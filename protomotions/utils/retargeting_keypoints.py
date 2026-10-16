# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conventions of the retargeting keypoint files.

data/scripts/keypoint_utils.py writes the files and the PyRoki retargeters
(pyroki/batch_retarget_to_*_from_keypoints.py) read them. Keypoint 0 is the
pelvis, 1 and 2 the left and right hip joints.
"""

from __future__ import annotations

import numpy as np

# Where keypoint_utils stores the SOMA pelvis keypoint, in the Hips joint frame
# (whose -y is forward): 7 cm behind the joint. The value was carried over from
# the Bones v24 skeleton, whose spine leaves the hips going backward. Robots match
# the pelvis keypoint with their pelvis origin, so the retargeters move it back
# onto the joint (soma_pelvis_keypoint_on_hips); keypoint files already on disk
# all carry the shift.
SOMA_PELVIS_KEYPOINT_SHIFT = (0.0, 0.07, 0.0)

# Forward position of the midpoint of the two SOMA hip joints, in the Hips frame:
# 2.6 cm in front of the Hips joint (soma23_humanoid.xml). It identifies which
# convention a keypoint file uses.
_SOMA_HIP_MIDPOINT_Y = -0.026
_TOLERANCE = 0.015


def _hip_midpoint_y(positions: np.ndarray, orientations: np.ndarray) -> float:
    """Median forward position of the hip midpoint in the pelvis keypoint frame."""
    hip_mid = 0.5 * (positions[:, 1] + positions[:, 2]) - positions[:, 0]
    local = np.einsum("tji,tj->ti", orientations[:, 0], hip_mid)
    return float(np.median(local[:, 1]))


def soma_pelvis_keypoint_on_hips(
    positions: np.ndarray, orientations: np.ndarray
) -> np.ndarray:
    """SOMA keypoint positions with the pelvis keypoint on the Hips joint.

    Files written with the shift are moved back by it; files whose pelvis
    keypoint is already on the joint are returned unchanged. Anything else (for
    example keypoints of another skeleton) raises ValueError. Only keypoint 0
    moves: the pelvis aux point keeps its stored position, 15.5 cm in front of
    the joint.

    Args:
        positions: [T, K, 3] keypoint positions.
        orientations: [T, K, 3, 3] keypoint orientations (keypoint 0 = Hips).
    """
    shift = np.asarray(SOMA_PELVIS_KEYPOINT_SHIFT, dtype=positions.dtype)
    hip_mid_y = _hip_midpoint_y(positions, orientations)
    if abs(hip_mid_y - (_SOMA_HIP_MIDPOINT_Y - shift[1])) < _TOLERANCE:
        positions = positions.copy()
        positions[:, 0] -= np.einsum("tij,j->ti", orientations[:, 0], shift)
        return positions
    if abs(hip_mid_y - _SOMA_HIP_MIDPOINT_Y) < _TOLERANCE:
        return positions
    raise ValueError(
        "Not SOMA keypoints: the hip joints are "
        f"{-hip_mid_y * 100:.1f} cm in front of the pelvis keypoint; expected "
        f"{(shift[1] - _SOMA_HIP_MIDPOINT_Y) * 100:.1f} (extracted with the pelvis "
        f"shift) or {-_SOMA_HIP_MIDPOINT_Y * 100:.1f} (without it)."
    )
