# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the rotation-conversion helpers in protomotions.utils.rotations
beyond the basic quaternion ops covered by test_rotations.py.

Targets vec_to_heading / heading_to_quat, calc_heading / calc_heading_quat,
slerp, quat_diff_norm, quat_to/from euler, quat_to/from exp_map,
quat_to/from tan_norm, axis_angle conversions, normalize_angle, and
angle_from_matrix_axis.
"""

from __future__ import annotations

import math

import pytest
import torch

from protomotions.utils.rotations import (
    angle_axis_to_exp_map,
    angle_from_matrix_axis,
    axis_angle_to_quaternion,
    calc_heading,
    calc_heading_quat,
    calc_heading_quat_inv,
    exp_map_to_angle_axis,
    exp_map_to_quat,
    get_basis_vector,
    heading_to_quat,
    matrix_to_quaternion,
    normalize_angle,
    quat_apply,
    quat_apply_yaw,
    quat_axis,
    quat_diff_norm,
    quat_diff_rad,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_identity,
    quat_mul,
    quat_normalize,
    quat_to_exp_map,
    quat_to_tan_norm,
    quaternion_to_matrix,
    slerp,
    tan_norm_to_quat,
    vec_to_heading,
)


def _rot_z(angle: float) -> torch.Tensor:
    """Single (1, 4) xyzw quaternion for rotation about +Z."""
    return quat_from_angle_axis(
        torch.tensor([angle]),
        torch.tensor([[0.0, 0.0, 1.0]]),
        w_last=True,
    )


# ---------- vec_to_heading / heading_to_quat -----------------------------------


def test_vec_to_heading_returns_atan2_y_x_in_radians():
    vecs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    headings = vec_to_heading(vecs)
    expected = torch.tensor([0.0, math.pi / 2, math.pi, -math.pi / 2])
    assert torch.allclose(headings, expected, atol=1e-6)


def test_heading_to_quat_round_trips_via_calc_heading_for_xyzw():
    angles = torch.tensor([0.5, -0.25, math.pi / 3])
    q = heading_to_quat(angles, w_last=True)
    recovered = calc_heading(q, w_last=True)
    assert torch.allclose(recovered, angles, atol=1e-5)


# ---------- calc_heading / heading quat ----------------------------------------


def test_calc_heading_for_pure_z_rotation_matches_rotation_angle():
    """For a pure z-axis rotation, the heading equals the rotation angle."""
    q = _rot_z(math.pi / 4)
    assert torch.allclose(
        calc_heading(q, w_last=True),
        torch.tensor([math.pi / 4]),
        atol=1e-5,
    )


def test_calc_heading_quat_inv_undoes_calc_heading_quat():
    q = _rot_z(math.pi / 3)
    heading_q = calc_heading_quat(q, w_last=True)
    inv_heading_q = calc_heading_quat_inv(q, w_last=True)
    # Combining the two should yield identity (within float tolerance).
    combined = quat_mul(heading_q, inv_heading_q, w_last=True)
    expected = quat_identity([1], w_last=True)
    assert torch.allclose(combined.abs(), expected.abs(), atol=1e-5)


def test_quat_apply_yaw_is_equivalent_to_applying_heading_quat():
    q = _rot_z(math.pi / 6)
    v = torch.tensor([[1.0, 0.0, 0.0]])
    yaw_applied = quat_apply_yaw(q, v, w_last=True)
    heading_q = calc_heading_quat(q, w_last=True)
    direct = quat_apply(heading_q, v, w_last=True)
    assert torch.allclose(yaw_applied, direct, atol=1e-5)


# ---------- slerp --------------------------------------------------------------


def test_slerp_at_endpoints_returns_input_quaternions():
    q0 = _rot_z(0.0)
    q1 = _rot_z(math.pi / 2)
    t0 = torch.tensor([[0.0]])
    t1 = torch.tensor([[1.0]])
    assert torch.allclose(slerp(q0, q1, t0), q0, atol=1e-5)
    assert torch.allclose(slerp(q0, q1, t1), q1, atol=1e-5)


def test_slerp_at_midpoint_equals_half_angle_rotation():
    q0 = _rot_z(0.0)
    q1 = _rot_z(math.pi / 2)
    half = slerp(q0, q1, torch.tensor([[0.5]]))
    expected = _rot_z(math.pi / 4)
    # Quaternions are equivalent up to sign — compare absolute values.
    assert torch.allclose(half.abs(), expected.abs(), atol=1e-5)


def test_slerp_takes_short_path_when_dot_product_negative():
    """When the dot product is negative, slerp should flip q1 so it always
    interpolates the shorter geodesic. We verify by feeding -q (which is the
    same rotation as q) and confirming slerp produces the same midpoint as
    with +q."""
    q0 = _rot_z(0.0)
    q1 = _rot_z(math.pi / 2)
    q1_neg = -q1
    t = torch.tensor([[0.5]])
    out_pos = slerp(q0, q1, t)
    out_neg = slerp(q0, q1_neg, t)
    assert torch.allclose(out_pos.abs(), out_neg.abs(), atol=1e-5)


# ---------- quat_diff_rad / quat_diff_norm -------------------------------------


def test_quat_diff_rad_zero_for_identical_quats():
    q = _rot_z(math.pi / 5)
    diff = quat_diff_rad(q, q.clone(), w_last=True)
    assert torch.allclose(diff, torch.zeros_like(diff), atol=1e-5)


def test_quat_diff_norm_returns_smaller_of_two_directions():
    """The minimum-of-two-directions clip means the diff is in [0, π]."""
    q0 = _rot_z(0.0)
    q1 = _rot_z(0.9 * math.pi)  # near maximum geodesic distance
    diff = quat_diff_norm(q0, q1, w_last=True)
    assert (diff >= 0).all()
    assert (diff <= math.pi).all()


# ---------- normalize_angle ----------------------------------------------------


def test_normalize_angle_wraps_to_minus_pi_pi_range():
    angles = torch.tensor([3 * math.pi, -3 * math.pi, math.pi / 2, -math.pi / 2])
    normalized = normalize_angle(angles)
    # normalize_angle uses atan2(sin, cos) so the output is in [-π, π].
    assert torch.all(normalized >= -math.pi - 1e-6)
    assert torch.all(normalized <= math.pi + 1e-6)
    # ±π/2 should be unchanged.
    assert math.isclose(normalized[2].item(), math.pi / 2, abs_tol=1e-6)
    assert math.isclose(normalized[3].item(), -math.pi / 2, abs_tol=1e-6)


# ---------- get_basis_vector / quat_axis ---------------------------------------


def test_get_basis_vector_is_quat_rotate_alias():
    q = _rot_z(math.pi / 3)
    v = torch.tensor([[1.0, 0.0, 0.0]])
    assert torch.allclose(
        get_basis_vector(q, v, w_last=True),
        quat_apply(q, v, w_last=True),
        atol=1e-5,
    )


# ---------- quaternion_to_matrix / matrix_to_quaternion ------------------------


def test_quaternion_to_matrix_identity_returns_identity_matrix():
    q = quat_identity([1], w_last=True)
    mat = quaternion_to_matrix(q, w_last=True)
    assert torch.allclose(mat, torch.eye(3).unsqueeze(0), atol=1e-6)


def test_quaternion_to_matrix_round_trips_through_matrix_to_quaternion():
    q = quat_normalize(torch.tensor([[0.1, 0.2, 0.3, 0.9]]))
    mat = quaternion_to_matrix(q, w_last=True)
    back = matrix_to_quaternion(mat, w_last=True)
    # Quaternions equivalent up to sign.
    assert torch.allclose(back.abs(), q.abs(), atol=1e-4)


def test_matrix_to_quaternion_z90_matches_known_quaternion():
    half = math.pi / 4
    expected = torch.tensor([[0.0, 0.0, math.sin(half), math.cos(half)]])
    rot_mat_z90 = torch.tensor([
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    ])
    q = matrix_to_quaternion(rot_mat_z90, w_last=True)
    assert torch.allclose(q.abs(), expected.abs(), atol=1e-5)


# ---------- axis_angle_to_quaternion / exp_map ---------------------------------


def test_axis_angle_to_quaternion_z90_matches_quat_from_angle_axis():
    # axis_angle vector encodes angle * axis (3,) format.
    axis_angle = torch.tensor([[0.0, 0.0, math.pi / 2]])
    q_from_aa = axis_angle_to_quaternion(axis_angle, w_last=True)
    q_from_helper = quat_from_angle_axis(
        torch.tensor([math.pi / 2]),
        torch.tensor([[0.0, 0.0, 1.0]]),
        w_last=True,
    )
    assert torch.allclose(q_from_aa.abs(), q_from_helper.abs(), atol=1e-5)


def test_exp_map_to_quat_round_trips_via_quat_to_exp_map():
    exp = torch.tensor([[0.1, 0.2, 0.3]])
    q = exp_map_to_quat(exp, w_last=True)
    recovered = quat_to_exp_map(q, w_last=True)
    assert torch.allclose(recovered, exp, atol=1e-5)


def test_angle_axis_to_exp_map_multiplies_angle_by_axis():
    angle = torch.tensor([math.pi / 3])
    axis = torch.tensor([[0.0, 1.0, 0.0]])
    exp = angle_axis_to_exp_map(angle, axis)
    expected = torch.tensor([[0.0, math.pi / 3, 0.0]])
    assert torch.allclose(exp, expected, atol=1e-6)


def test_exp_map_to_angle_axis_unit_axis_for_pure_z():
    exp = torch.tensor([[0.0, 0.0, math.pi / 4]])
    angle, axis = exp_map_to_angle_axis(exp)
    assert torch.allclose(angle, torch.tensor([math.pi / 4]), atol=1e-5)
    assert torch.allclose(axis, torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-5)


def test_exp_map_to_angle_axis_zero_input_returns_zero_angle_default_axis():
    exp = torch.tensor([[0.0, 0.0, 0.0]])
    angle, axis = exp_map_to_angle_axis(exp)
    assert math.isclose(angle.item(), 0.0, abs_tol=1e-6)
    # Default axis is (0, 0, 1) for the small-angle fallback path.
    assert torch.allclose(axis, torch.tensor([[0.0, 0.0, 1.0]]))


# ---------- euler ↔ quat -------------------------------------------------------


def test_quat_from_euler_xyz_zero_returns_identity():
    z = torch.tensor([0.0])
    q = quat_from_euler_xyz(z, z, z, w_last=True)
    expected = quat_identity([1], w_last=True)
    assert torch.allclose(q.abs(), expected.abs(), atol=1e-6)


def test_quat_from_euler_xyz_pure_yaw_matches_z_rotation():
    yaw = torch.tensor([math.pi / 4])
    zero = torch.tensor([0.0])
    q = quat_from_euler_xyz(zero, zero, yaw, w_last=True)
    expected = _rot_z(math.pi / 4)
    assert torch.allclose(q.abs(), expected.abs(), atol=1e-6)


# ---------- tan_norm ↔ quat ----------------------------------------------------


def test_quat_to_tan_norm_then_back_recovers_quaternion():
    q = quat_normalize(torch.tensor([[0.2, 0.3, 0.4, 0.8]]))
    tan_norm = quat_to_tan_norm(q, w_last=True)
    assert tan_norm.shape == (1, 6)
    # Tangent is q * x_hat, normal is q * z_hat — both unit length.
    assert torch.allclose(tan_norm[..., :3].norm(dim=-1), torch.tensor([1.0]), atol=1e-5)
    assert torch.allclose(tan_norm[..., 3:].norm(dim=-1), torch.tensor([1.0]), atol=1e-5)

    recovered = tan_norm_to_quat(tan_norm, w_last=True)
    # Round-trip equivalence up to quaternion sign.
    assert torch.allclose(recovered.abs(), q.abs(), atol=1e-4)


# ---------- angle_from_matrix_axis ---------------------------------------------


def test_angle_from_matrix_axis_extracts_z_angle_from_z_rotation_matrix():
    angles = [math.pi / 6, -math.pi / 4, math.pi / 3]
    rot_mats = torch.stack(
        [quaternion_to_matrix(_rot_z(a), w_last=True)[0] for a in angles]
    )
    axis = torch.tensor([0.0, 0.0, 1.0])
    extracted = angle_from_matrix_axis(rot_mats, axis)
    assert torch.allclose(extracted, torch.tensor(angles), atol=1e-5)


def test_angle_from_matrix_axis_zero_for_identity():
    eye = torch.eye(3).unsqueeze(0)
    angle = angle_from_matrix_axis(eye, torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(angle, torch.zeros(1), atol=1e-6)


# ---------- quat_axis ----------------------------------------------------------


def test_quat_axis_z90_swaps_basis_vectors_correctly():
    q = _rot_z(math.pi / 2)
    # +X axis after 90° z rotation → +Y.
    x_axis = quat_axis(q, axis=0, w_last=True)
    assert torch.allclose(x_axis, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-5)
    # +Y axis after 90° z rotation → -X.
    y_axis = quat_axis(q, axis=1, w_last=True)
    assert torch.allclose(y_axis, torch.tensor([[-1.0, 0.0, 0.0]]), atol=1e-5)
    # +Z is unchanged.
    z_axis = quat_axis(q, axis=2, w_last=True)
    assert torch.allclose(z_axis, torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-5)
