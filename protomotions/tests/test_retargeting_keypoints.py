# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from protomotions.utils.retargeting_keypoints import (
    SOMA_PELVIS_KEYPOINT_SHIFT,
    soma_pelvis_keypoint_on_hips,
)


# Every keypoint file on disk was written with this shift.
SHIFT_ON_DISK = 0.07


def _random_rotations(rng, shape):
    q = rng.normal(size=(*shape, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            np.stack(
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1
            ),
            np.stack(
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1
            ),
            np.stack(
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1
            ),
        ],
        -2,
    )


def _soma_keypoints(pelvis_shift_y, frames=40, seed=0):
    """Pelvis keypoint, two hip joints and one more keypoint for randomly posed
    Hips frames; the hips sit 2.6 cm in front of the Hips joint (local -y). The
    other keypoints get their own orientations, as in real files."""
    rng = np.random.default_rng(seed)
    R = _random_rotations(rng, (frames,))
    hips = rng.normal(size=(frames, 3))
    local = np.array(
        [
            [0, pelvis_shift_y, 0],
            [0.09, -0.026, -0.05],
            [-0.09, -0.026, -0.05],
            [0, -0.2, 0.3],
        ]
    )
    positions = hips[:, None] + np.einsum("tij,kj->tki", R, local)
    others = _random_rotations(rng, (frames, len(local) - 1))
    orientations = np.concatenate([R[:, None], others], axis=1)
    return positions.astype(np.float32), orientations.astype(np.float32), hips


def test_shift_matches_the_files_on_disk():
    assert SOMA_PELVIS_KEYPOINT_SHIFT == (0.0, SHIFT_ON_DISK, 0.0)


def test_shifted_pelvis_keypoint_moves_onto_the_hips_joint():
    positions, orientations, hips = _soma_keypoints(SHIFT_ON_DISK)
    before = positions.copy()
    out = soma_pelvis_keypoint_on_hips(positions, orientations)
    np.testing.assert_allclose(out[:, 0], hips, atol=1e-5)
    np.testing.assert_array_equal(out[:, 1:], positions[:, 1:])
    np.testing.assert_array_equal(positions, before)  # input left untouched


def test_unshifted_keypoints_are_unchanged():
    positions, orientations, _ = _soma_keypoints(0.0)
    out = soma_pelvis_keypoint_on_hips(positions, orientations)
    np.testing.assert_array_equal(out, positions)
    # Applying it twice is harmless.
    shifted, orientations, _ = _soma_keypoints(SHIFT_ON_DISK)
    once = soma_pelvis_keypoint_on_hips(shifted, orientations)
    np.testing.assert_array_equal(
        soma_pelvis_keypoint_on_hips(once, orientations), once
    )


def test_other_skeletons_are_rejected():
    # Bones v24 keypoint files put the hips 7 cm in front of the pelvis keypoint.
    positions, orientations, _ = _soma_keypoints(0.044)
    with pytest.raises(ValueError, match="Not SOMA keypoints"):
        soma_pelvis_keypoint_on_hips(positions, orientations)
