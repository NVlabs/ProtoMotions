# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""State derivation utilities for deployment.

These functions bridge the gap between raw simulator state and the derived
inputs the ONNX model expects.  They exist because the current observation
functions (`compute_humanoid_reduced_coords_observations`,
`build_deploy_target_poses`) accept pre-computed *derived* values
(``anchor_rot``, ``root_local_ang_vel``) rather than raw rigid-body arrays.

IMPORTANT body-index convention
--------------------------------
``anchor_rot``          -- uses the **ANCHOR** body (``torso_link``, index 16
                           for G1).  This is the IMU body on the physical robot.
``root_local_ang_vel``  -- uses the **ROOT** body   (``pelvis``, index 0).
                           These are DIFFERENT bodies; mixing them silently
                           produces wrong observations.

``root_local_ang_vel`` frame convention
-----------------------------------------
The ONNX model expects angular velocity in the **root body's local frame**.
Different sources provide angular velocity in different frames:

=========================  ==============  =================================
Source                     Frame           What to do
=========================  ==============  =================================
MuJoCo ``data.cvel``       World frame     Apply ``quat_rotate_inverse`` via
                                           ``compute_root_local_ang_vel_np``
MuJoCo ``data.qvel[3:6]``  Local frame     Use directly -- **no** rotation
Real robot IMU gyroscope    Local frame     Use directly -- **no** rotation
=========================  ==============  =================================

The ``compute_root_local_ang_vel_np`` / ``compute_root_local_ang_vel``
functions convert FROM world frame TO local frame.  If your source is
already local (``qvel``, IMU gyro), pass the value through unchanged.

TODO(future refactor): fold these derivations INTO the observation functions
so they accept ``rigid_body_rot`` / ``rigid_body_ang_vel`` arrays plus static
body-index parameters, eliminating the need for any pre-computation by the
deployer.  See ``protomotions/envs/obs/humanoid.py``
``compute_humanoid_reduced_coords_observations``.

Quaternion convention
---------------------
All functions in this module use the **xyzw** convention (ProtoMotions
common format).  MuJoCo provides quaternions in **wxyz** format; use
``mujoco_wxyz_to_xyzw`` to convert at the read boundary.

MuJoCo state-array layout reminder
------------------------------------
- ``data.qpos[7:]``        -- DOF positions   (7-dof free-joint prefix)
- ``data.qvel[6:]``        -- DOF velocities  (6-dof free-joint prefix)
- ``data.xquat[body+1]``   -- body quaternion (wxyz); index +1 because the
                               world body occupies index 0
- ``data.cvel[body+1]``    -- body velocity ``[ang_vel(3), lin_vel(3)]``
                               angular velocity is the first three elements
                               **in WORLD frame** (not body local)
- ``data.qvel[3:6]``       -- free-joint angular velocity
                               **in BODY LOCAL frame** (not world)
"""

from __future__ import annotations

import numpy as np

__all__ = [
    # NumPy versions -- used in cached-mode deploy (no PyTorch computation)
    "mujoco_wxyz_to_xyzw",
    "compute_anchor_rot_np",
    "compute_root_local_ang_vel_np",
    # Heading alignment (NumPy)
    "compute_yaw_offset_np",
    "apply_heading_offset_np",
    # PyTorch versions -- used during ONNX export / first-run deploy
    "compute_anchor_rot",
    "compute_root_local_ang_vel",
]

# ---------------------------------------------------------------------------
# NumPy helpers (no PyTorch required)
# ---------------------------------------------------------------------------


def mujoco_wxyz_to_xyzw(wxyz: np.ndarray) -> np.ndarray:
    """Convert a MuJoCo quaternion (wxyz) to ProtoMotions xyzw convention.

    Works on any array whose last dimension is 4.

    Args:
        wxyz: Quaternion(s) in wxyz order, shape ``(..., 4)``.

    Returns:
        Quaternion(s) in xyzw order, same shape.
    """
    return wxyz[..., [1, 2, 3, 0]]


def compute_anchor_rot_np(
    rigid_body_rot: np.ndarray,
    anchor_body_index: int,
) -> np.ndarray:
    """Extract the anchor body's orientation from the full body-rotation array.

    During deployment this corresponds to the **torso IMU** quaternion.

    Args:
        rigid_body_rot: Body orientations, shape ``[num_bodies, 4]`` (xyzw).
        anchor_body_index: 0-based index of the anchor body (e.g. 16 for
            ``torso_link`` on the G1).

    Returns:
        Anchor body orientation, shape ``[4,]`` (xyzw).
    """
    return rigid_body_rot[anchor_body_index]


def compute_root_local_ang_vel_np(
    rigid_body_rot: np.ndarray,
    rigid_body_ang_vel: np.ndarray,
    root_body_index: int = 0,
) -> np.ndarray:
    """Convert root angular velocity from WORLD frame to LOCAL frame (NumPy).

    Implements ``quat_rotate_inverse(root_rot, root_ang_vel)`` -- rotates the
    world-frame angular velocity into the root body's local frame.

    USE THIS ONLY when your source angular velocity is in WORLD frame
    (e.g., MuJoCo ``data.cvel[body+1, 0:3]``).

    DO NOT USE when your source is already in LOCAL frame:

    - MuJoCo ``data.qvel[3:6]`` for a free joint is ALREADY local-frame.
    - Real robot IMU gyroscope reads in body-local frame.

    In those cases, use the angular velocity directly as ``root_local_ang_vel``
    without any rotation.  Applying this function to already-local data
    double-rotates the vector, producing wrong values that only become
    apparent during turns (when body heading diverges from world axes).

    IMPORTANT: uses the **ROOT** body (pelvis, index 0), NOT the anchor body.

    Args:
        rigid_body_rot: Body orientations, shape ``[num_bodies, 4]`` (xyzw).
        rigid_body_ang_vel: Body angular velocities, shape
            ``[num_bodies, 3]`` (**world frame**).
        root_body_index: 0-based index of the root body (default 0 = pelvis).

    Returns:
        Root angular velocity in local frame, shape ``[3,]``.
    """
    root_rot = rigid_body_rot[root_body_index]      # [4] xyzw
    root_ang_vel = rigid_body_ang_vel[root_body_index]  # [3]
    return _quat_rotate_inverse_np(root_rot, root_ang_vel)


def _quat_rotate_inverse_np(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Pure-NumPy equivalent of ``rotations.quat_rotate_inverse`` (xyzw convention).

    Rotates vector ``v`` by the *inverse* of quaternion ``q``.
    Equivalent to expressing ``v`` in the body frame defined by ``q``.

    Formula (matches the TorchScript implementation in rotations.py)::

        a = v * (2 * q_w^2 - 1)
        b = cross(q_vec, v) * q_w * 2
        c = q_vec * dot(q_vec, v) * 2
        result = a - b + c

    Args:
        q_xyzw: Unit quaternion ``[x, y, z, w]``, shape ``[4,]``.
        v: Vector in world frame, shape ``[3,]``.

    Returns:
        Vector in body frame, shape ``[3,]``.
    """
    q_w = q_xyzw[3]
    q_vec = q_xyzw[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


# ---------------------------------------------------------------------------
# Heading alignment (NumPy) -- for real-robot deployment
# ---------------------------------------------------------------------------


def _extract_yaw_quat_np(q_xyzw: np.ndarray) -> np.ndarray:
    """Extract the yaw-only quaternion from a full orientation (xyzw).

    Decomposes the orientation into yaw (rotation about world Z) and
    returns a quaternion that represents only that yaw.

    Args:
        q_xyzw: Unit quaternion ``[x, y, z, w]``, shape ``[4,]``.

    Returns:
        Yaw-only quaternion ``[x, y, z, w]``, shape ``[4,]``.
    """
    # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    x, y, z, w = q_xyzw[0], q_xyzw[1], q_xyzw[2], q_xyzw[3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    # Yaw-only quaternion: rotation about Z by `yaw`
    half = yaw * 0.5
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float32)


def _quat_mul_np(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> np.ndarray:
    """Hamilton product of two xyzw quaternions (pure NumPy).

    Works on single quaternions ``[4,]`` or batched ``[..., 4]``.
    """
    ax, ay, az, aw = a_xyzw[..., 0], a_xyzw[..., 1], a_xyzw[..., 2], a_xyzw[..., 3]
    bx, by, bz, bw = b_xyzw[..., 0], b_xyzw[..., 1], b_xyzw[..., 2], b_xyzw[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1).astype(np.float32)


def _quat_conjugate_np(q_xyzw: np.ndarray) -> np.ndarray:
    """Conjugate (inverse for unit quats) of an xyzw quaternion."""
    result = q_xyzw.copy()
    result[..., :3] *= -1.0
    return result


def compute_yaw_offset_np(
    robot_quat_xyzw: np.ndarray,
    motion_quat_xyzw: np.ndarray,
) -> np.ndarray:
    """Compute a yaw-only heading offset between robot and motion frames.

    Returns a quaternion ``R_offset`` such that::

        R_offset * motion_body_rot ≈ aligned_body_rot

    where ``aligned_body_rot`` is in the robot's heading frame.

    At ``t=0``, this makes ``quat_inv(robot_anchor) * R_offset * motion_anchor``
    near-identity, which is the correct initial condition for the obs function.

    In simulation (robot starts at motion frame 0), the offset is near-identity
    and has no effect.  On the real robot, it compensates for arbitrary startup
    heading.

    Args:
        robot_quat_xyzw:  Robot anchor body orientation at t=0, ``[4,]`` (xyzw).
        motion_quat_xyzw: Motion anchor body orientation at frame 0, ``[4,]`` (xyzw).

    Returns:
        Yaw-only offset quaternion ``[4,]`` (xyzw).
    """
    robot_yaw = _extract_yaw_quat_np(robot_quat_xyzw)
    motion_yaw = _extract_yaw_quat_np(motion_quat_xyzw)
    # offset = yaw(robot) * yaw(motion)^-1
    return _quat_mul_np(robot_yaw, _quat_conjugate_np(motion_yaw))


def apply_heading_offset_np(
    offset_quat_xyzw: np.ndarray,
    body_rots_xyzw: np.ndarray,
) -> np.ndarray:
    """Apply a heading offset to an array of body rotations.

    Computes ``offset * body_rot`` for every quaternion in the array.

    Args:
        offset_quat_xyzw: Yaw-only offset ``[4,]`` (xyzw), from
            :func:`compute_yaw_offset_np`.
        body_rots_xyzw: Body rotations, shape ``[..., 4]`` (xyzw).
            Typically ``[num_future_steps, num_bodies, 4]``.

    Returns:
        Aligned body rotations, same shape as input.
    """
    original_shape = body_rots_xyzw.shape
    flat = body_rots_xyzw.reshape(-1, 4)
    # Broadcast offset [4,] against flat [N, 4]
    offset_broadcast = np.broadcast_to(offset_quat_xyzw, flat.shape)
    aligned = _quat_mul_np(offset_broadcast, flat)
    return aligned.reshape(original_shape)


# ---------------------------------------------------------------------------
# PyTorch helpers (used during ONNX export and first-run deploy)
# ---------------------------------------------------------------------------


def compute_anchor_rot(rigid_body_rot, anchor_body_index: int):
    """Extract the anchor body's orientation from the full body-rotation tensor.

    PyTorch version of :func:`compute_anchor_rot_np`.

    Args:
        rigid_body_rot: Body orientations ``[num_envs, num_bodies, 4]`` (xyzw).
        anchor_body_index: 0-based index of the anchor body.

    Returns:
        Anchor body orientation ``[num_envs, 4]`` (xyzw).
    """
    return rigid_body_rot[:, anchor_body_index, :]


def compute_root_local_ang_vel(
    rigid_body_rot,
    rigid_body_ang_vel,
    root_body_index: int = 0,
    w_last: bool = True,
):
    """Convert root angular velocity from WORLD frame to LOCAL frame (PyTorch).

    PyTorch version of :func:`compute_root_local_ang_vel_np`.
    See that function's docstring for when to use vs. when to pass through.

    IMPORTANT: uses the **ROOT** body (pelvis), NOT the anchor body.

    Args:
        rigid_body_rot: Body orientations ``[num_envs, num_bodies, 4]`` (xyzw).
        rigid_body_ang_vel: Body angular velocities
            ``[num_envs, num_bodies, 3]`` (**world frame**).
        root_body_index: 0-based index of the root body (default 0 = pelvis).
        w_last: Quaternion convention; ``True`` = xyzw (ProtoMotions default).

    Returns:
        Root angular velocity in local frame ``[num_envs, 3]``.
    """
    from protomotions.utils.rotations import quat_rotate_inverse

    root_rot = rigid_body_rot[:, root_body_index, :]
    root_ang_vel = rigid_body_ang_vel[:, root_body_index, :]
    return quat_rotate_inverse(root_rot, root_ang_vel, w_last)
