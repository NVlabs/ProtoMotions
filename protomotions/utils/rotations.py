# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
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
import math
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple, List
import unittest


@torch.jit.script
def normalize(x, eps: float = 1e-9):
    """Normalize vector along last dimension."""
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def wxyz_to_xyzw(quat: Tensor):
    shape = quat.shape
    flat_quat = quat.reshape(-1, 4)
    flat_quat = flat_quat[:, [1, 2, 3, 0]]
    return flat_quat.reshape(shape)


@torch.jit.script
def xyzw_to_wxyz(quat: Tensor):
    shape = quat.shape
    flat_quat = quat.reshape(-1, 4)
    flat_quat = flat_quat[:, [3, 0, 1, 2]]
    return flat_quat.reshape(shape)


@torch.jit.script
def _sqrt_positive_part(x: Tensor) -> Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


@torch.jit.script
def quat_mul(a, b, w_last: bool):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    if w_last:
        x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    else:
        w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    if w_last:
        quat = torch.stack([x, y, z, w], dim=-1).reshape(shape)
    else:
        quat = torch.stack([w, x, y, z], dim=-1).reshape(shape)

    return quat


@torch.jit.script
def quat_identity(shape: List[int], w_last: bool) -> Tensor:
    """
    Construct 3D identity rotation given shape
    """
    q = torch.zeros(shape + [4])
    if w_last:
        q[..., 3] = 1
    else:
        q[..., 0] = 1
    return q


@torch.jit.script
def quat_identity_like(x, w_last: bool) -> Tensor:
    """
    Construct identity 3D rotation with the same shape
    """
    return quat_identity(x.shape[:-1], w_last).to(x.device, x.dtype)


@torch.jit.script
def quat_conjugate(a: Tensor, w_last: bool) -> Tensor:
    shape = a.shape
    a = a.reshape(-1, 4)
    if w_last:
        return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).reshape(shape)
    else:
        return torch.cat((a[:, 0:1], -a[:, 1:]), dim=-1).reshape(shape)


@torch.jit.script
def quat_apply(a: Tensor, b: Tensor, w_last: bool) -> Tensor:
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    if w_last:
        xyz = a[:, :3]
        w = a[:, 3:]
    else:
        xyz = a[:, 1:]
        w = a[:, :1]
    t = xyz.cross(b, dim=-1) * 2
    return (b + w * t + xyz.cross(t, dim=-1)).reshape(shape)


@torch.jit.script
def quat_rotate(q: Tensor, v: Tensor, w_last: bool) -> Tensor:
    shape = q.shape
    flat_q = q.reshape(-1, shape[-1])
    flat_v = v.reshape(-1, v.shape[-1])
    if w_last:
        q_w = flat_q[:, -1]
        q_vec = flat_q[:, :3]
    else:
        q_w = flat_q[:, 0]
        q_vec = flat_q[:, 1:]
    a = flat_v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, flat_v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = (
        q_vec
        * torch.bmm(
            q_vec.reshape(flat_q.shape[0], 1, 3), flat_v.reshape(flat_q.shape[0], 3, 1)
        ).squeeze(-1)
        * 2.0
    )
    return (a + b + c).reshape(v.shape)


@torch.jit.script
def quat_rotate_inverse(q: Tensor, v: Tensor, w_last: bool) -> Tensor:
    shape = q.shape
    if w_last:
        q_w = q[:, -1]
        q_vec = q[:, :3]
    else:
        q_w = q[:, 0]
        q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = (
        q_vec
        * torch.bmm(q_vec.reshape(shape[0], 1, 3), v.reshape(shape[0], 3, 1)).squeeze(
            -1
        )
        * 2.0
    )
    return a - b + c


@torch.jit.script
def quat_unit(a):
    return normalize(a)


@torch.jit.script
def quat_pos(x: Tensor, w_last: bool = True) -> Tensor:
    """
    make all the real part of the quaternion positive
    """
    q = x
    if w_last:
        w = q[..., 3:]
    else:
        w = q[..., 0:1]
    z = (w < 0).float()
    q = (1 - 2 * z) * q
    return q


@torch.jit.script
def quat_normalize(q):
    """
    Construct 3D rotation from quaternion (the quaternion needs not to be normalized).
    """
    q = quat_unit(quat_pos(q))  # normalized to positive and unit quaternion
    return q


@torch.jit.script
def quat_mul_norm(x: Tensor, y: Tensor, w_last: bool) -> Tensor:
    """
    Combine two set of 3D rotations together using \**\* operator. The shape needs to be
    broadcastable
    """
    return quat_normalize(quat_mul(x, y, w_last))


@torch.jit.script
def quat_angle_axis(x: Tensor, w_last: bool) -> Tuple[Tensor, Tensor]:
    """
    The (angle, axis) representation of the rotation. The axis is normalized to unit length.
    The angle is guaranteed to be between [0, pi].

    Use atan2 instead of arccos for stability.
    See the simpler arccos implementation in unit test.
    """
    shape = x.shape[:-1]
    quat = x.reshape(-1, 4)

    # Ensure scalar part is non-negative for shortest angle [0, pi]
    scalar_index = 3 if w_last else 0
    needs_flip = quat[..., scalar_index] < 0
    quat = torch.where(needs_flip.unsqueeze(-1), -quat, quat)

    if w_last:
        w = quat[..., 3]
        axis = quat[..., :3]
    else:
        w = quat[..., 0]
        axis = quat[..., 1:]

    norm_axis = torch.norm(axis, p=2, dim=-1)
    angle = 2 * torch.atan2(norm_axis, w)
    axis_normalized = axis / norm_axis.unsqueeze(-1).clamp(min=1e-9)

    # Reshape output
    angle = angle.reshape(shape)
    axis_normalized = axis_normalized.reshape(shape + (3,))

    return angle, axis_normalized


@torch.jit.script
def quat_from_angle_axis(angle: Tensor, axis: Tensor, w_last: bool) -> Tensor:
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    if w_last:
        return quat_unit(torch.cat([xyz, w], dim=-1))
    else:
        return quat_unit(torch.cat([w, xyz], dim=-1))


@torch.jit.script
def vec_to_heading(h_vec):
    h_theta = torch.atan2(h_vec[..., 1], h_vec[..., 0])
    return h_theta


@torch.jit.script
def heading_to_quat(h_theta, w_last: bool):
    axis = torch.zeros(
        h_theta.shape
        + [
            3,
        ],
        device=h_theta.device,
    )
    axis[..., 2] = 1
    heading_q = quat_from_angle_axis(h_theta, axis, w_last=w_last)
    return heading_q


@torch.jit.script
def quat_axis(q: Tensor, axis: int, w_last: bool) -> Tensor:
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec, w_last)


@torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


@torch.jit.script
def get_basis_vector(q: Tensor, v: Tensor, w_last: bool) -> Tensor:
    return quat_rotate(q, v, w_last)


@torch.jit.script
def get_euler_xyz(q: Tensor, w_last: bool) -> Tuple[Tensor, Tensor, Tensor]:
    if w_last:
        qx, qy, qz, qw = 0, 1, 2, 3
    else:
        qw, qx, qy, qz = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = (
        q[:, qw] * q[:, qw]
        - q[:, qx] * q[:, qx]
        - q[:, qy] * q[:, qy]
        + q[:, qz] * q[:, qz]
    )
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    half_pi = torch.tensor(torch.pi / 2.0, device=q.device, dtype=q.dtype)
    pitch = torch.where(
        torch.abs(sinp) >= 1, torch.copysign(half_pi, sinp), torch.asin(sinp)
    )

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = (
        q[:, qw] * q[:, qw]
        + q[:, qx] * q[:, qx]
        - q[:, qy] * q[:, qy]
        - q[:, qz] * q[:, qz]
    )
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2 * torch.pi), pitch % (2 * torch.pi), yaw % (2 * torch.pi)


@torch.jit.script
def quat_from_euler_xyz(
    roll: Tensor, pitch: Tensor, yaw: Tensor, w_last: bool
) -> Tensor:
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    if w_last:
        return torch.stack([qx, qy, qz, qw], dim=-1)
    else:
        return torch.stack([qw, qx, qy, qz], dim=-1)


@torch.jit.script
def quat_diff_rad(a: Tensor, b: Tensor, w_last: bool) -> Tensor:
    """
    Get the difference in radians between two quaternions.

    Args:
        a: first quaternion, shape (N, 4)
        b: second quaternion, shape (N, 4)
    Returns:
        Difference in radians, shape (N,)
    """
    b_conj = quat_conjugate(b, w_last)
    mul = quat_mul(a, b_conj, w_last)
    # 2 * torch.acos(torch.abs(mul[:, -1]))
    return 2.0 * torch.asin(torch.clamp(torch.norm(mul[:, 1:], p=2, dim=-1), max=1.0))


@torch.jit.script
def quat_apply_yaw(quat: Tensor, vec: Tensor, w_last: bool) -> Tensor:
    quat_yaw = quat.clone().reshape(-1, 4)
    quat_yaw[:, :2] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec, w_last)


@torch.jit.script
def quaternion_to_matrix(quaternions: torch.Tensor, w_last: bool) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions of shape (..., 4).
        w_last: If True, the real part of the quaternion is last.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if w_last:
        i, j, k, r = torch.unbind(quaternions, -1)
    else:
        r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


@torch.jit.script
def axis_angle_to_quaternion(axis_angle: torch.Tensor, w_last: bool) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
        w_last: If True, the real part of the quaternion is last.

    Returns:
        quaternions as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    if w_last:
        quaternions = wxyz_to_xyzw(quaternions)
    return quaternions


@torch.jit.script
def quat_from_two_vectors(
    v1: torch.Tensor, v2: torch.Tensor, w_last: bool, eps: float = 1e-6
) -> torch.Tensor:
    """Calculate minimal rotation quaternion that rotates vector a to vector b.

    Args:
        a: Source vectors of shape (..., 3)
        b: Target vectors of shape (..., 3)
        w_last: If True, quaternion format is (x,y,z,w), else (w,x,y,z)

    Returns:
        Quaternion rotations of shape (..., 4)
    """

    orig_shape = v1.shape
    v1 = v1.reshape(-1, 3)
    v2 = v2.reshape(-1, 3)
    dot = (v1 * v2).sum(-1)
    cross = torch.cross(v1, v2, dim=-1)
    out = torch.cat([(1 + dot).unsqueeze(-1), cross], dim=-1)
    # handle v1 & v2 with same direction
    sind = dot > 1 - eps
    out[sind] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=v1.device)
    # handle v1 & v2 with opposite direction
    nind = dot < -1 + eps
    if torch.any(nind):
        vx = torch.tensor([1.0, 0.0, 0.0], device=v1.device)
        vxdot = (v1 * vx).sum(-1).abs()
        nxind = nind & (vxdot < 1 - eps)
        if torch.any(nxind):
            out[nxind] = axis_angle_to_quaternion(
                normalize(torch.cross(vx.expand_as(v1[nxind]), v1[nxind], dim=-1)),
                w_last=w_last,
            )
        # handle v1 & v2 with opposite direction and they are parallel to x axis
        pind = nind & (vxdot >= 1 - eps)
        if torch.any(pind):
            vy = torch.tensor([0.0, 1.0, 0.0], device=v1.device)
            out[pind] = axis_angle_to_quaternion(
                normalize(torch.cross(vy.expand_as(v1[pind]), v1[pind], dim=-1))
                * math.pi,
                w_last=w_last,
            )
    # normalize and reshape
    out = normalize(out).reshape(orig_shape[:-1] + (4,))

    if w_last:
        return wxyz_to_xyzw(out)
    else:
        return out


@torch.jit.script
def matrix_to_quaternion(matrix: torch.Tensor, w_last: bool) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        w_last: If True, the real part of the quaternion is last.

    Returns:
        quaternions as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    quat_candidates = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(batch_dim + (4,))

    if w_last:
        quat_candidates = wxyz_to_xyzw(quat_candidates)

    return quat_candidates


@torch.jit.script
def quat_to_angle_axis(q: Tensor, w_last: bool = False) -> Tuple[Tensor, Tensor]:
    # computes axis-angle representation from quaternion q
    # q must be normalized
    min_theta = 1e-5
    if not w_last:
        qx, _, qz, qw = 1, 2, 3, 0
    else:
        qx, _, qz, qw = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx : qz + 1] / sin_theta_expand

    mask = torch.abs(sin_theta) > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis


@torch.jit.script
def angle_axis_to_exp_map(angle: Tensor, axis: Tensor) -> Tensor:
    # compute exponential map from axis-angle
    angle_expand = angle.unsqueeze(-1)
    exp_map = angle_expand * axis
    return exp_map


@torch.jit.script
def quat_to_exp_map(q: Tensor, w_last: bool = False) -> Tensor:
    # compute exponential map from quaternion
    # q must be normalized
    angle, axis = quat_to_angle_axis(q, w_last)
    exp_map = angle_axis_to_exp_map(angle, axis)
    return exp_map


@torch.jit.script
def quat_to_tan_norm(q: Tensor, w_last: bool) -> Tensor:
    # represents a rotation using the tangent and normal vectors
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    tan = quat_rotate(q, ref_tan, w_last)

    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    norm = quat_rotate(q, ref_norm, w_last)

    norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return norm_tan


@torch.jit.script
def angle_from_matrix_axis(rot_mat: Tensor, axis: Tensor) -> Tensor:
    """
    Extracts the rotation angle around a specified axis from a batch of rotation matrices.

    Args:
        rot_mat (B, 3, 3): Batch of rotation matrices.
        axis (3,): The axis of rotation. Should be normalized.

    Returns:
        torch.Tensor (B,): Batch of rotation angles in radians.
    """
    axis = axis.to(
        rot_mat.device, rot_mat.dtype
    )  # Ensure axis is on correct device/dtype

    # cos(theta) = (trace(R) - 1) / 2
    cos_theta = (torch.einsum("bii->b", rot_mat) - 1.0) / 2.0
    # Clamp for numerical stability
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    # sin(theta) * axis = 0.5 * (R - R^T)_vec
    # where _vec extracts components (R32-R23, R13-R31, R21-R12)
    skew_sym = 0.5 * (rot_mat - rot_mat.transpose(-1, -2))
    sin_theta_axis = torch.stack(
        [skew_sym[:, 2, 1], skew_sym[:, 0, 2], skew_sym[:, 1, 0]], dim=-1
    )  # Shape (B, 3)

    # Project sin_theta_axis onto the known axis to get sin_theta
    # sin_theta = dot(sin_theta_axis, axis)
    sin_theta = torch.einsum("bi,i->b", sin_theta_axis, axis)

    # Compute angle using atan2
    angle = torch.atan2(sin_theta, cos_theta)
    return angle


@torch.jit.script
def exp_map_to_angle_axis(exp_map: Tensor) -> Tuple[Tensor, Tensor]:
    min_theta = 1e-5

    angle = torch.norm(exp_map, dim=-1)
    angle_exp = torch.unsqueeze(angle, dim=-1)
    axis = exp_map / angle_exp
    angle = normalize_angle(angle)

    default_axis = torch.zeros_like(exp_map)
    default_axis[..., -1] = 1

    mask = torch.abs(angle) > min_theta
    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)

    return angle, axis


@torch.jit.script
def exp_map_to_quat(exp_map: Tensor, w_last: bool) -> Tensor:
    angle, axis = exp_map_to_angle_axis(exp_map)
    q = quat_from_angle_axis(angle, axis, w_last)
    return q


@torch.jit.script
def calc_heading(q: Tensor, w_last: bool) -> Tensor:
    # calculate heading direction from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = quat_rotate(q, ref_dir, w_last)

    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading


@torch.jit.script
def calc_heading_quat(q: Tensor, w_last: bool) -> Tensor:
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q, w_last)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(heading, axis, w_last)
    return heading_q


@torch.jit.script
def calc_heading_quat_inv(q: Tensor, w_last: bool = False) -> Tensor:
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q, w_last)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(-heading, axis, w_last)
    return heading_q


@torch.jit.script
def slerp(q0: Tensor, q1: Tensor, t: Tensor) -> Tensor:
    cos_half_theta = torch.sum(q0 * q1, dim=-1)

    neg_mask = cos_half_theta < 0
    q1 = q1.clone()
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratioB = torch.sin(t * half_theta) / sin_half_theta

    new_q = ratioA * q0 + ratioB * q1

    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)

    return new_q


@torch.jit.script
def quat_diff_norm(quat1: Tensor, quat2: Tensor, w_last: bool):
    if w_last:
        w = 3
    else:
        w = 0
    quat1inv = quat_conjugate(quat1, w_last)
    mul = quat_mul(quat1inv, quat2, w_last)
    norm = mul[..., w].clip(-1, 1).arccos() * 2
    # Trying both rotation directions
    norm = torch.min(norm, math.pi * 2 - norm)
    return norm


@torch.jit.script
def quat_angle_diff_norm(quat1: Tensor, quat2: Tensor, w_last: bool):
    diff_quat = quat_mul(quat2, quat_conjugate(quat1, w_last), w_last)
    angle_axis = quat_to_angle_axis(diff_quat, w_last)[0]
    return angle_axis**2


# Define the updated test cases
class TestQuatFromTwoVectors(unittest.TestCase):
    def test_same_vectors(self):
        a = torch.tensor([[1.0, 0.0, 0.0]])
        b = torch.tensor([[1.0, 0.0, 0.0]])
        expected = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        result = quat_from_two_vectors(a, b, w_last=True)
        print("result 1: ", result)
        torch.testing.assert_close(result, expected)

    # def test_opposite_vectors(self):
    #     a = torch.tensor([[1.0, 0.0, 0.0]])
    #     b = torch.tensor([[-1.0, 0.0, 0.0]])
    #     expected = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    #     result = quat_from_two_vectors(a, b, w_last=True)
    #     print("result 2: ", result)
    #     torch.testing.assert_close(result, expected)

    def test_orthogonal_vectors(self):
        a = torch.tensor([[1.0, 0.0, 0.0]])
        b = torch.tensor([[0.0, 1.0, 0.0]])

        expected = torch.tensor([[0.0, 0.0, 0.7071, 0.7071]])
        result = quat_from_two_vectors(a, b, w_last=True)
        print("result 3: ", result)
        torch.testing.assert_close(result, expected)

    def test_w_first_format(self):
        a = torch.tensor([[1.0, 0.0, 0.0]])
        b = torch.tensor([[0.0, 1.0, 0.0]])
        # 90 around z
        expected = torch.tensor([[0.7071, 0.0, 0.0, 0.7071]])
        result = quat_from_two_vectors(a, b, w_last=False)
        print("result 4: ", result)
        torch.testing.assert_close(result, expected)


class TestQuatAngleAxis(unittest.TestCase):
    def alternative_quat_angle_axis(self, x, w_last):
        """Alternative implementation of quat_angle_axis to compare against"""
        if w_last:
            w = x[..., -1]
            axis = x[..., :3]
        else:
            w = x[..., 0]
            axis = x[..., 1:]
        s = 2 * (w**2) - 1
        angle = s.clamp(-1, 1).arccos()  # just to be safe
        axis = axis / axis.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9)
        return angle, axis

    def test_implementations_comparison(self):
        # Test with various quaternions
        test_quats = [
            # Identity quaternion
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
            # 90 degree rotation around X
            torch.tensor([0.7071068, 0.0, 0.0, 0.7071068]),
            # 180 degree rotation around Y
            torch.tensor([0.0, 1.0, 0.0, 0.0]),
            # 45 degree rotation around Z
            torch.tensor([0.0, 0.0, 0.3826834, 0.9238795]),
            # Random quaternion
            normalize(torch.tensor([0.1, 0.2, 0.3, 0.4])),
        ]

        for q in test_quats:
            q = q.unsqueeze(0)  # Add batch dimension

            # Test with w_last=True
            angle1, axis1 = quat_angle_axis(q, w_last=True)
            angle2, axis2 = self.alternative_quat_angle_axis(q, w_last=True)

            print(
                f"Current impl: angle={angle1.item():.4f}, axis={axis1.squeeze().tolist()}"
            )
            print(
                f"Alternative: angle={angle2.item():.4f}, axis={axis2.squeeze().tolist()}"
            )

            # Check if angles are close
            torch.testing.assert_close(angle1, angle2, rtol=1e-4, atol=1e-4)

            # For angles near 0, the axis can be arbitrary, so only check non-zero angles
            if angle1.item() > 1e-4:
                # Since axis and -axis represent the same rotation, we need to check both possibilities
                try:
                    torch.testing.assert_close(axis1, axis2, rtol=1e-4, atol=1e-4)
                except AssertionError:
                    torch.testing.assert_close(axis1, -axis2, rtol=1e-4, atol=1e-4)

            # Test with w_last=False
            q_wxyz = xyzw_to_wxyz(q)
            angle1, axis1 = quat_angle_axis(q_wxyz, w_last=False)
            angle2, axis2 = self.alternative_quat_angle_axis(q_wxyz, w_last=False)

            # Check if angles are close
            torch.testing.assert_close(angle1, angle2, rtol=1e-4, atol=1e-4)

            # For angles near 0, the axis can be arbitrary, so only check non-zero angles
            if angle1.item() > 1e-4:
                # Since axis and -axis represent the same rotation, we need to check both possibilities
                try:
                    torch.testing.assert_close(axis1, axis2, rtol=1e-4, atol=1e-4)
                except AssertionError:
                    torch.testing.assert_close(axis1, -axis2, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    # Run the unit tests
    unittest.TextTestRunner().run(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestQuatFromTwoVectors)
    )
    unittest.TextTestRunner().run(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestQuatAngleAxis)
    )
