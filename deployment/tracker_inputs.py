# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared ONNX input protocol for ProtoMotions tracker deployment.

This module contains framework-agnostic NumPy logic used by standalone MuJoCo
deployment and hardware integrations such as RoboJuDo. Hardware/framework code
is still responsible for reading current robot sensors; this module owns the
frame conventions and semantic ONNX input assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from deployment.state_utils import (
    apply_heading_offset_np,
    compute_heading_quat_inv_np,
    compute_yaw_offset_np,
    quat_rotate_np,
)


@dataclass
class TrackerInputRequirements:
    """Semantic ONNX inputs requested by an exported tracker."""

    odom_from_start: bool
    odom_start_xy: bool
    future_pos: bool
    future_anchor_pos: bool
    start_heading: bool
    ref_rigid_body_pos: bool
    ref_anchor_pos: bool

    @property
    def needs_reference_positions(self) -> bool:
        """True if the policy consumes reference *positions* in any form.

        Both the full body array and the anchor-only slice come from the same
        source array, so either one means the deployment has to supply and
        align reference positions.
        """
        return self.future_pos or self.future_anchor_pos

    @property
    def needs_spatial_start(self) -> bool:
        return (
            self.odom_from_start
            or self.odom_start_xy
            or self.needs_reference_positions
            or self.start_heading
            or self.ref_rigid_body_pos
            or self.ref_anchor_pos
        )


@dataclass
class TrackerAlignment:
    """Per-motion-start spatial alignment state."""

    heading_offset: np.ndarray
    odom_start_xy: np.ndarray
    odom_start_heading_inv: np.ndarray
    robot_anchor_start_pos: np.ndarray
    motion_anchor_start_pos: np.ndarray


def get_tracker_input_requirements(
    onnx_name_to_key: Dict[str, str],
) -> TrackerInputRequirements:
    """Inspect semantic keys to determine which deploy inputs are required."""
    semantic_keys = set(onnx_name_to_key.values())
    return TrackerInputRequirements(
        odom_from_start="odom_disp_start_corrupt" in semantic_keys,
        odom_start_xy="odom_start_xy" in semantic_keys,
        future_pos="mimic.future_pos" in semantic_keys,
        future_anchor_pos="mimic.future_anchor_pos" in semantic_keys,
        start_heading="odom_start_heading_inv" in semantic_keys,
        ref_rigid_body_pos="mimic.ref_state.rigid_body_pos" in semantic_keys,
        ref_anchor_pos="mimic.ref_anchor_pos" in semantic_keys,
    )


def init_tracker_alignment(
    robot_anchor_pos: np.ndarray,
    robot_anchor_rot: np.ndarray,
    motion_anchor_pos: np.ndarray,
    motion_anchor_rot: np.ndarray,
) -> TrackerAlignment:
    """Capture the shared spatial frame at motion start."""
    return TrackerAlignment(
        heading_offset=compute_yaw_offset_np(robot_anchor_rot, motion_anchor_rot),
        odom_start_xy=np.asarray(robot_anchor_pos[:2], dtype=np.float32).copy(),
        odom_start_heading_inv=compute_heading_quat_inv_np(robot_anchor_rot),
        robot_anchor_start_pos=np.asarray(robot_anchor_pos, dtype=np.float32).copy(),
        motion_anchor_start_pos=np.asarray(motion_anchor_pos, dtype=np.float32).copy(),
    )


def align_future_motion_refs(
    future_refs: dict,
    alignment: TrackerAlignment,
) -> dict:
    """Move the reference motion from its recorded frame into the robot's start frame.

    Every entry present in ``future_refs`` is aligned. Aligning positions is not
    optional: ``mimic.future_anchor_pos`` is sliced out of ``body_pos``, and the
    policy graph subtracts ``mimic.ref_anchor_pos`` from it to recover how far the
    reference itself travels. ``mimic.ref_anchor_pos`` is always expressed in the
    robot's frame, because the odometer offset differences it against the robot's
    own believed position. Leaving ``body_pos`` in the motion's recorded frame
    therefore makes that subtraction return the offset between the two frames --
    typically tens of metres, since clips live wherever they were captured --
    rather than the reference's travel.

    Keys the caller does not supply are left absent, so a deployment whose policy
    reads no reference positions does not have to provide them.
    """
    aligned = dict(future_refs)
    if "body_rot" in future_refs:
        aligned["body_rot"] = apply_heading_offset_np(
            alignment.heading_offset, future_refs["body_rot"]
        )
    if "body_pos" in future_refs:
        motion_disp = future_refs["body_pos"] - alignment.motion_anchor_start_pos
        aligned["body_pos"] = alignment.robot_anchor_start_pos + quat_rotate_np(
            alignment.heading_offset, motion_disp
        )
    return aligned


def compute_odom_disp_start(
    robot_anchor_xy: np.ndarray,
    alignment: TrackerAlignment,
    corruption_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """Compute start-heading-frame odometer displacement from robot anchor XY."""
    robot_disp_world = (
        np.asarray(robot_anchor_xy, dtype=np.float32) - alignment.odom_start_xy
    )
    robot_disp_world_3d = np.array(
        [robot_disp_world[0], robot_disp_world[1], 0.0], dtype=np.float32
    )
    odom_disp_start = quat_rotate_np(
        alignment.odom_start_heading_inv, robot_disp_world_3d
    )[:2]
    if corruption_fn is not None:
        odom_disp_start = corruption_fn(odom_disp_start)
    return odom_disp_start.astype(np.float32)


def build_tracker_onnx_inputs(
    *,
    onnx_input_names: list[str],
    onnx_name_to_key: Dict[str, str],
    dof_pos: np.ndarray,
    dof_vel: np.ndarray,
    anchor_rot: np.ndarray,
    root_local_ang_vel: np.ndarray,
    future_refs: dict,
    anchor_body_index: int,
    num_dofs: int,
    prev_actions: Optional[np.ndarray] = None,
    odom_start_xy: Optional[np.ndarray] = None,
    odom_disp_start: Optional[np.ndarray] = None,
    odom_start_heading_inv: Optional[np.ndarray] = None,
    ref_rigid_body_pos: Optional[np.ndarray] = None,
    ref_anchor_pos: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Assemble actual ONNX input arrays from semantic deploy tensors."""
    if prev_actions is None:
        prev_actions = np.zeros(num_dofs, dtype=np.float32)

    key_to_array = {
        "current.dof_pos": dof_pos[None],
        "current.dof_vel": dof_vel[None],
        "current.anchor_rot": anchor_rot[None],
        "current.root_local_ang_vel": root_local_ang_vel[None],
        "historical.processed_actions": prev_actions[None, None],
    }
    # Only source what the caller actually supplied. A deployment whose policy
    # reads no reference positions has no reason to carry the full body array.
    if "body_rot" in future_refs:
        key_to_array["mimic.future_rot"] = future_refs["body_rot"][None]
        key_to_array["mimic.future_anchor_rot"] = future_refs["body_rot"][
            :, anchor_body_index, :
        ][None]
    if "body_pos" in future_refs:
        key_to_array["mimic.future_pos"] = future_refs["body_pos"][None]
        key_to_array["mimic.future_anchor_pos"] = future_refs["body_pos"][
            :, anchor_body_index, :
        ][None]
    if "dof_pos" in future_refs:
        key_to_array["mimic.future_dof_pos"] = future_refs["dof_pos"][None]
    if "dof_vel" in future_refs:
        key_to_array["mimic.future_dof_vel"] = future_refs["dof_vel"][None]
    if odom_start_xy is not None:
        key_to_array["odom_start_xy"] = odom_start_xy[None]
    if odom_disp_start is not None:
        key_to_array["odom_disp_start_corrupt"] = odom_disp_start[None]
    if odom_start_heading_inv is not None:
        key_to_array["odom_start_heading_inv"] = odom_start_heading_inv[None]
    if ref_anchor_pos is not None:
        # Reference anchor position at the current step. With
        # mimic.future_anchor_pos this tells the tracker how far the reference
        # itself moves, and it is the reference point for the odometer offset.
        key_to_array["mimic.ref_anchor_pos"] = ref_anchor_pos[None]
    if ref_rigid_body_pos is not None:
        # Full reference body array: only needed by observations that read every
        # reference body. The baked-FK tracker path does NOT -- it needs the
        # anchor alone -- so most deployments can leave this out.
        key_to_array["mimic.ref_state.rigid_body_pos"] = ref_rigid_body_pos[None]
        key_to_array.setdefault(
            "mimic.ref_anchor_pos", ref_rigid_body_pos[anchor_body_index][None]
        )

    onnx_inputs = {}
    unsourced = []
    for onnx_name in onnx_input_names:
        semantic_key = onnx_name_to_key.get(onnx_name)
        if semantic_key not in key_to_array:
            unsourced.append(f"{onnx_name} ({semantic_key})")
            continue
        value = key_to_array[semantic_key]
        onnx_inputs[onnx_name] = (
            value.astype(np.bool_)
            if value.dtype == np.bool_
            else value.astype(np.float32)
        )
    if unsourced:
        # Skipping quietly here just defers the failure to onnxruntime, which
        # reports a missing feed without saying which semantic quantity is
        # absent or who was supposed to provide it.
        raise KeyError(
            "The exported policy requires inputs this assembler cannot source: "
            + ", ".join(unsourced)
            + ". Supply the missing reference data in `future_refs` or the "
            "optional arguments, or extend `build_tracker_onnx_inputs` to "
            "cover the semantic key."
        )
    return onnx_inputs
