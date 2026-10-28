# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The current G1 scene policy's sensors and incoming-action history.

Positions are metres, angular velocities radians/second, and rotations xyzw.
The graph reconstructs body state from joints and the torso IMU. Scene object
poses, when available, must share the localization frame with the root pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


CONTRACT = "protomotions.masked_mimic.g1.v5_raw_context_scene"
HISTORY_LENGTH = 8
INPUT_SHAPES = {
    "current.root_pos": (3,),
    "current.root_rot": (4,),
    "masked_mimic.anchor_rotation_mask": (5, 2),
    "masked_mimic.anchor_translation_mask": (5, 2),
    "masked_mimic.average_velocity": (2,),
    "masked_mimic.local_ref_pos": (5, 33, 3),
    "masked_mimic.local_ref_rot": (5, 33, 4),
    "masked_mimic.local_target_bodies_masks": (50,),
    "masked_mimic.local_target_poses_masks": (5,),
    "masked_mimic.local_time_offsets": (5,),
    "masked_mimic.ref_pos": (5, 33, 3),
    "masked_mimic.ref_rot": (5, 33, 4),
    "masked_mimic.style_condition_masks": (5, 11),
    "masked_mimic.style_frame_masks": (5,),
    "masked_mimic.style_pose_samples": (225,),
    "masked_mimic.style_velocity_samples": (150,),
    "masked_mimic.target_bodies_masks": (50,),
    "masked_mimic.target_poses_masks": (5,),
    "masked_mimic.time_offsets": (5,),
    "masked_mimic.velocity_command_mask": (1,),
    "noisy.anchor_pos": (3,),
    "noisy.anchor_rot": (4,),
    "noisy.dof_pos": (29,),
    "noisy.dof_vel": (29,),
    "noisy.root_local_ang_vel": (3,),
    "noisy_historical.anchor_rot": (8, 4),
    "noisy_historical.dof_pos": (8, 29),
    "noisy_historical.dof_vel": (8, 29),
    "noisy_historical.raw_actions": (8, 29),
    "noisy_historical.root_local_ang_vel": (8, 3),
    "raw_action_history": (9, 29),
    "scene_boxes.centers": (1, 2, 3),
    "scene_boxes.extents": (1, 2, 3),
    "scene_boxes.object_pos": (1, 3),
    "scene_boxes.object_rot": (1, 4),
    "scene_boxes.position_offset": (3,),
    "scene_boxes.valid": (1, 2),
    "scene_boxes.visible": (1, 1),
    "scene_boxes.yaw_offset": (),
}
BOOL_INPUTS = frozenset(
    key for key in INPUT_SHAPES if key.endswith(("mask", "masks"))
) | {"scene_boxes.valid", "scene_boxes.visible"}


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Reject changed input semantics before constructing the runtime."""
    if manifest["contract"] != CONTRACT:
        raise ValueError("expected the G1 scene raw-context contract")
    keys = manifest["input_keys"]
    if len(keys) != len(INPUT_SHAPES) or set(keys) != set(INPUT_SHAPES):
        raise ValueError("scene policy input keys differ from the reviewed contract")
    columns = [manifest[key] for key in ("input_names", "input_shapes", "input_dtypes")]
    if any(len(column) != len(keys) for column in columns):
        raise ValueError("scene input metadata lengths differ")
    for key, name, shape, dtype in zip(keys, *columns):
        expected_dtype = "bool" if key in BOOL_INPUTS else "float32"
        if (
            name != key.replace(".", "_")
            or tuple(shape) != INPUT_SHAPES[key]
            or dtype != expected_dtype
        ):
            raise ValueError(
                f"scene input {key!r} has an unsupported name, shape or dtype"
            )
    robot = manifest["robot_contract"]
    if (
        robot["quaternion_order"] != "xyzw"
        or robot["anchor_body_index"] != 16
        or robot["conditionable_body_ids"] != [16, 13, 7, 24, 32]
        or len(robot["body_names"]) != 33
        or len(robot["joint_names"]) != 29
    ):
        raise ValueError("scene policy robot layout differs from the G1 contract")
    # Goal and recovery kernels use this reviewed G1 ordering. Shape equality
    # alone would accept a renamed/reordered rig with different semantics.
    expected_bodies = {
        7: "left_ankle_roll_link",
        13: "right_ankle_roll_link",
        16: "torso_link",
        24: "left_rubber_hand",
        32: "right_rubber_hand",
    }
    if any(robot["body_names"][i] != name for i, name in expected_bodies.items()):
        raise ValueError("scene policy recovery body ordering differs from G1")
    if (
        robot["joint_names"][3] != "left_knee_joint"
        or robot["joint_names"][9] != "right_knee_joint"
    ):
        raise ValueError("scene policy recovery knee ordering differs from G1")
    if manifest["previous_action_domain"] != "raw_policy_action":
        raise ValueError("scene policy requires raw incoming actions")
    action_map = manifest["action_map"]
    if action_map["kind"] != "affine":
        raise ValueError("scene policy requires an invertible affine action map")
    scale, offset = (
        np.asarray(action_map[key], np.float32) for key in ("scale", "offset")
    )
    if (
        scale.shape != (29,)
        or offset.shape != (29,)
        or not np.isfinite(scale).all()
        or not np.isfinite(offset).all()
        or np.any(scale == 0)
    ):
        raise ValueError(
            "scene action scale/offset must be finite 29-vectors with nonzero scales"
        )


@dataclass(frozen=True)
class SensorReading:
    """One synchronized joint/IMU/localization sample; no simulator body state."""

    dof_pos: np.ndarray
    dof_vel: np.ndarray
    root_local_ang_vel: np.ndarray
    anchor_rot: np.ndarray
    anchor_pos: np.ndarray
    root_pos: np.ndarray
    root_rot: np.ndarray

    def copied(self) -> "SensorReading":
        return SensorReading(
            *(
                np.array(value, dtype=np.float32, copy=True)
                for value in (
                    self.dof_pos,
                    self.dof_vel,
                    self.root_local_ang_vel,
                    self.anchor_rot,
                    self.anchor_pos,
                    self.root_pos,
                    self.root_rot,
                )
            )
        )


class IncomingActionHistory:
    """At s(t), expose a(t-1) and eight pairs (s(t-k), a(t-k-1)).

    Preparing inference never advances history. Commit the pre-step reading
    and *accepted* raw action once after transport/physics accepts the command.
    Reset repeats the first measured state and fills all incoming actions zero.
    """

    def __init__(self) -> None:
        self.states: list[SensorReading] = []
        self.actions = np.zeros((HISTORY_LENGTH + 1, 29), dtype=np.float32)

    def reset(self) -> None:
        self.states.clear()
        self.actions.fill(0)

    def inputs(self, current: SensorReading) -> dict[str, np.ndarray]:
        if not self.states:
            self.states = [current.copied() for _ in range(HISTORY_LENGTH)]
        # Recorders keep the inference inputs after transport acknowledgement.
        # Advancing history must not mutate the inputs that produced that action.
        actions = self.actions.copy()
        values = {
            "current.root_pos": current.root_pos,
            "current.root_rot": current.root_rot,
            "noisy.anchor_pos": current.anchor_pos,
            "raw_action_history": actions,
            "noisy_historical.raw_actions": actions[1:],
        }
        for name in ("dof_pos", "dof_vel", "root_local_ang_vel", "anchor_rot"):
            values[f"noisy.{name}"] = getattr(current, name)
            values[f"noisy_historical.{name}"] = np.stack(
                [getattr(row, name) for row in self.states]
            )
        return values

    def commit(self, before_step: SensorReading, raw_action: np.ndarray) -> None:
        action = np.asarray(raw_action, dtype=np.float32)
        if action.shape != (29,) or not np.isfinite(action).all():
            raise ValueError("accepted raw action must be a finite 29-vector")
        if not self.states:
            raise RuntimeError("prepare policy inputs before committing a command")
        self.states.insert(0, before_step.copied())
        del self.states[HISTORY_LENGTH:]
        self.actions[1:] = self.actions[:-1].copy()
        self.actions[0] = action


def hidden_scene_inputs() -> dict[str, np.ndarray]:
    """No object localization: zero visibility/validity, identity orientation."""
    values = {
        key: np.zeros(shape, dtype=bool if key in BOOL_INPUTS else np.float32)
        for key, shape in INPUT_SHAPES.items()
        if key.startswith("scene_boxes.")
    }
    values["scene_boxes.object_rot"][0, 3] = 1
    return values
