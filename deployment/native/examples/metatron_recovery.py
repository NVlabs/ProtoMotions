# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared sensor-only recovery prompts for the G1 scene policy.

This changes conditioning, never motors. Height means torso origin above the
lowest FK body origin, not terrain clearance. Positions are metres, times
seconds and rotations xyzw. Keep phase thresholds aligned with the web demo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


OFFSETS = np.array([0.2, 0.4, 0.6, 0.8, 1.0], np.float32)
# Slow the plan's progression, not its future lookahead or arrival times.
PRONE_PLAN_RATE = 1 / 3
TUCKED_HAND_HALF_WIDTH = 0.12
# Torso height above the lowest FK body, in metres. A crouch needs upward
# goals, not the lying/seated templates reserved for poses near the floor.
FLOOR_POSE_HEIGHT = 0.35
# Source seconds in the bundled clip, after its belly-down preparation.
SEATED_PLAN_START_SECONDS = 4.5


def contains_inversion(torso_rotations: np.ndarray) -> bool:
    """A commanded roll owns its whole clip, including entry/exit transitions."""
    up = 1 - 2 * (torso_rotations[:, 0] ** 2 + torso_rotations[:, 1] ** 2)
    return bool(np.any(up < 0.25))


@dataclass(frozen=True)
class RecoveryPose:
    positions: np.ndarray
    rotations: np.ndarray
    joints: np.ndarray

    def sensed(self) -> tuple[float, float, float, float]:
        """Height proxy in metres, torso up/forward Z components, knees in radians."""
        x, y, z, w = self.rotations[16]
        height = float(self.positions[16, 2] - self.positions[:, 2].min())
        up, forward_up = 1 - 2 * (x * x + y * y), 2 * (x * z - w * y)
        knees = float((self.joints[3] + self.joints[9]) * 0.5)
        return height, float(up), float(forward_up), knees

    def right_hand_lateral(self) -> float:
        """Torso-local hand Y in metres; a tucked-arm proxy, not contact sensing."""
        x, y, z, w = self.rotations[16]
        torso_y = np.array(
            [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)]
        )
        return float(torso_y @ (self.positions[32] - self.positions[16]))


def slerp(a: np.ndarray, b: np.ndarray, fraction: float) -> np.ndarray:
    """Shortest-arc xyzw interpolation; sinc keeps coincident rotations exact."""
    dot = (a * b).sum(-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    angle = np.arccos(np.clip(np.abs(dot), 0, 1)) / np.pi
    result = (
        (1 - fraction) * np.sinc((1 - fraction) * angle) * a
        + fraction * np.sinc(fraction * angle) * b
    ) / np.sinc(angle)
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


class StagedRecovery:
    """One recovery owner after task goals, with no task-specific recovery state."""

    def __init__(self, plan: dict[str, Any], body_names: list[str]) -> None:
        if (
            plan["body_names"] != body_names
            or plan["anchor_body_index"] != 16
            or plan["quaternion_order"] != "xyzw"
        ):
            raise ValueError("recovery plan and policy body layout differ")
        self.positions = np.asarray(plan["positions"], np.float32)
        self.rotations = np.asarray(plan["rotations"], np.float32)
        self.dt = float(plan["dt_seconds"])
        frames = plan["frames"]
        if (
            frames < 2
            or not np.isfinite(self.dt)
            or self.dt <= 0
            or self.positions.shape != (frames, 33, 3)
            or self.rotations.shape != (frames, 33, 4)
            or not np.isfinite(self.positions).all()
            or not np.isfinite(self.rotations).all()
            or not np.allclose(np.linalg.norm(self.rotations, axis=-1), 1, atol=1e-4)
        ):
            raise ValueError(
                "recovery plan must contain finite poses and unit quaternions"
            )
        # Sagittal reflection swaps body identities as well as Y. For xyzw
        # quaternions the reflected rotation is (-x, y, -z, w).
        opposite = [
            name.replace("left_", "right_", 1)
            if name.startswith("left_")
            else name.replace("right_", "left_", 1)
            if name.startswith("right_")
            else name
            for name in body_names
        ]
        indices = [body_names.index(name) for name in opposite]
        self.mirrored_positions = self.positions[:, indices] * np.array(
            [1, -1, 1], np.float32
        )
        self.mirrored_rotations = self.rotations[:, indices] * np.array(
            [-1, 1, -1, 1], np.float32
        )
        self.reset()

    def reset(self) -> None:
        self.phase = "inactive"
        self.elapsed = 0.0
        self.phase_start = 0.0
        self.ready_since: float | None = None
        self.just_finished = False
        self.transitions: list[tuple[str, float]] = []
        self.mirror_prone = False

    def _transition(self, phase: str) -> None:
        self.phase = phase
        self.phase_start = self.elapsed
        self.transitions.append((phase, self.elapsed))

    def step(
        self, inputs: dict[str, np.ndarray], pose: RecoveryPose, dt: float
    ) -> bool:
        """Replace named, unbatched task inputs when measured geometry indicates a fall."""
        height, up, forward, knees = pose.sensed()
        fallen = height < 0.5 or up < 0.25
        self.just_finished = False
        if self.phase == "inactive" and (
            not fallen or self._intentional_low(inputs, height, up)
        ):
            return False
        if self.phase == "inactive" or (self.phase == "settle" and fallen):
            self.elapsed = 0.0
            self.ready_since = None
            if height < FLOOR_POSE_HEIGHT and forward < -0.65:
                phase = "prone_plan"
            elif height < FLOOR_POSE_HEIGHT and up > 0.3 and knees > 0.8:
                phase = "seated_plan"
            else:
                phase = "ramp"
            self._transition(phase)
            # Choose once on entry; changing handedness mid-plan would jump goals.
            self.mirror_prone = (
                phase == "prone_plan"
                and abs(pose.right_hand_lateral()) < TUCKED_HAND_HALF_WIDTH
            )
        else:
            self.elapsed += dt
        if self.phase != "settle":
            if height > 0.55 and up > 0.5:
                self.ready_since = (
                    self.elapsed if self.ready_since is None else self.ready_since
                )
            else:
                self.ready_since = None
            if (
                self.ready_since is not None
                and self.elapsed - self.ready_since >= 0.2 - 1e-9
            ):
                self._transition("settle")
        self._apply(inputs, height)
        if (
            self.phase == "settle"
            and self.elapsed - self.phase_start >= 0.5 - 1e-9
            and height > 0.7
            and up > 0.85
        ):
            self._transition("inactive")
            self.just_finished = True
        return True

    def _intentional_low(
        self, inputs: dict[str, np.ndarray], height: float, up: float
    ) -> bool:
        if up <= 0.3:
            return False
        requested = -np.inf
        style = inputs["masked_mimic.style_pose_samples"].reshape(5, 5, 9)
        visible = (
            inputs["masked_mimic.style_frame_masks"]
            & inputs["masked_mimic.style_condition_masks"][:, 0]
        )
        if visible.any():
            requested = float(style[visible, 0, 2].max())
        for prefix in ("", "local_"):
            positions = inputs[f"masked_mimic.{prefix}ref_pos"]
            visible = (
                inputs[f"masked_mimic.{prefix}target_bodies_masks"]
                .reshape(5, 5, 2)[:, 0, 0]
                .copy()
            )
            visible &= inputs[f"masked_mimic.{prefix}target_poses_masks"]
            if not prefix:
                visible &= inputs["masked_mimic.anchor_translation_mask"][:, 1]
            if visible.any():
                heights = positions[:, 16, 2] - positions[:, :, 2].min(-1)
                requested = max(requested, float(heights[visible].max()))
        return requested < 0.5 and abs(requested - height) < 0.15

    def _apply(self, inputs: dict[str, np.ndarray], height: float) -> None:
        for name in (
            "target_bodies_masks",
            "target_poses_masks",
            "anchor_translation_mask",
            "anchor_rotation_mask",
            "style_condition_masks",
            "style_frame_masks",
            "velocity_command_mask",
            "local_target_bodies_masks",
            "local_target_poses_masks",
        ):
            inputs[f"masked_mimic.{name}"].fill(0)
        inputs["scene_boxes.visible"].fill(0)
        masks = inputs["masked_mimic.local_target_bodies_masks"].reshape(5, 5, 2)
        inputs["masked_mimic.local_target_poses_masks"].fill(True)
        inputs["masked_mimic.local_time_offsets"][:] = OFFSETS
        positions = inputs["masked_mimic.local_ref_pos"]
        rotations = inputs["masked_mimic.local_ref_rot"]
        if self.phase == "prone_plan":
            self._sample_plan(positions, rotations)
            masks[:, :, 0] = True
            masks[:, 0, 1] = True
        elif self.phase == "seated_plan":
            self._sample_plan(positions, rotations, SEATED_PLAN_START_SECONDS)
            masks[:, :, 0] = True
            masks[:, 0, 1] = True
        else:
            positions.fill(0)
            rotations.fill(0)
            rotations[:, :, 3] = 1
            settling = self.phase == "settle"
            positions[:, 16, 2] = (
                0.79
                if settling
                else height + np.clip(0.79 - height, -0.8 * OFFSETS, 0.8 * OFFSETS)
            )
            masks[:, 0, 0] = True
            masks[:, 0, 1] = settling
            inputs["masked_mimic.average_velocity"].fill(0)
            inputs["masked_mimic.velocity_command_mask"][:] = settling

    def _sample_plan(
        self, positions: np.ndarray, rotations: np.ndarray, source_start: float = 0
    ) -> None:
        plan_seconds = (
            source_start + (self.elapsed - self.phase_start) * PRONE_PLAN_RATE
        )
        mirrored = self.phase == "prone_plan" and self.mirror_prone
        source_pos = self.mirrored_positions if mirrored else self.positions
        source_rot = self.mirrored_rotations if mirrored else self.rotations
        for slot, offset in enumerate(OFFSETS):
            frame = min(
                max((plan_seconds + float(offset)) / self.dt, 0),
                len(self.positions) - 1,
            )
            low, high = int(frame), min(int(frame) + 1, len(self.positions) - 1)
            fraction = frame - low
            positions[slot] = source_pos[low] + fraction * (
                source_pos[high] - source_pos[low]
            )
            rotations[slot] = slerp(source_rot[low], source_rot[high], fraction)
