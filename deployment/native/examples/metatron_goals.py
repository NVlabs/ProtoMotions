# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local body goals and hands-down idle for the current scene policy.

The exported graph removes target translation and heading from local body
poses. Only requested heading stays global; tracking must not pull the robot
back to the recorded trajectory's world position. Explicit odometry control
retains its position goal. All arrays here are unbatched graph inputs.
"""

from __future__ import annotations

import numpy as np


class MetatronGoals:
    def __init__(
        self,
        rest_positions: np.ndarray,
        rest_rotations: np.ndarray,
        rest_style: np.ndarray,
    ) -> None:
        self.positions = rest_positions.copy()
        self.rotations = rest_rotations.copy()
        self.style = rest_style.copy()

    def apply(self, inputs: dict[str, np.ndarray], mode: str, standing: bool) -> None:
        """Route task poses locally; optionally replace them with a static idle."""
        if mode == "odometry":
            return
        p = "masked_mimic."
        global_masks = inputs[p + "target_bodies_masks"].reshape(5, 5, 2)
        local_masks = inputs[p + "local_target_bodies_masks"].reshape(5, 5, 2)
        local_masks[:] = global_masks
        local_masks[:, 0, 0] &= inputs[p + "anchor_translation_mask"][:, 1]
        local_masks[:, 0, 1] &= inputs[p + "anchor_rotation_mask"][:, 0]
        inputs[p + "local_ref_pos"] = inputs[p + "ref_pos"].copy()
        inputs[p + "local_ref_rot"] = inputs[p + "ref_rot"].copy()
        inputs[p + "local_time_offsets"][:] = inputs[p + "time_offsets"]
        inputs[p + "local_target_poses_masks"][:] = inputs[
            p + "target_poses_masks"
        ] & local_masks.any(axis=(1, 2))
        heading = global_masks[:, 0, 1] & inputs[p + "anchor_rotation_mask"][:, 1]
        global_masks.fill(False)
        global_masks[:, 0, 1] = heading
        inputs[p + "anchor_translation_mask"].fill(False)
        inputs[p + "anchor_rotation_mask"][:, 0] = False
        inputs[p + "target_poses_masks"] &= heading
        if standing:
            self._idle(inputs)

    def _idle(self, inputs: dict[str, np.ndarray]) -> None:
        """Repeated local pose and zero-velocity style; no global position hold."""
        p = "masked_mimic."
        inputs[p + "local_ref_pos"][:] = self.positions
        inputs[p + "local_ref_rot"][:] = self.rotations
        masks = inputs[p + "local_target_bodies_masks"].reshape(5, 5, 2)
        masks.fill(False)
        masks[:, :, 0] = True
        masks[:, 0, 1] = True
        inputs[p + "local_target_poses_masks"].fill(True)
        inputs[p + "local_time_offsets"][:] = [0.1, 0.2, 0.3, 0.4, 0.5]
        inputs[p + "style_pose_samples"][:] = self.style
        inputs[p + "style_velocity_samples"].fill(0)
        inputs[p + "style_condition_masks"].fill(True)
        inputs[p + "style_frame_masks"].fill(True)
        inputs[p + "average_velocity"].fill(0)
        inputs[p + "velocity_command_mask"].fill(True)
