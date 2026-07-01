# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for quaternion utilities in protomotions.utils.rotations."""

import math

import torch

from protomotions.utils.rotations import (
    normalize,
    quat_angle_axis,
    quat_apply,
    quat_axis,
    quat_conjugate,
    quat_from_angle_axis,
    quat_from_two_vectors,
    quat_identity,
    quat_identity_like,
    quat_apply_yaw,
    quat_mul,
    quat_normalize,
    quat_pos,
    quat_rotate,
    quat_rotate_inverse,
    quat_unit,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


def test_wxyz_xyzw_round_trip_preserves_values():
    quat_wxyz = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    assert torch.equal(xyzw_to_wxyz(wxyz_to_xyzw(quat_wxyz)), quat_wxyz)
    quat_xyzw = torch.tensor([[2.0, 3.0, 4.0, 1.0]])
    assert torch.equal(wxyz_to_xyzw(xyzw_to_wxyz(quat_xyzw)), quat_xyzw)


def test_wxyz_to_xyzw_moves_real_part_to_last_axis():
    quat_wxyz = torch.tensor([[0.5, 1.0, 2.0, 3.0]])
    converted = wxyz_to_xyzw(quat_wxyz)
    assert torch.equal(converted, torch.tensor([[1.0, 2.0, 3.0, 0.5]]))


def test_normalize_returns_unit_norm_vectors_and_handles_zero_input():
    values = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    normalized = normalize(values)
    assert torch.allclose(normalized[0].norm(), torch.tensor(1.0), atol=1e-6)
    # Zero vector with eps clamp returns the original vector divided by eps,
    # which produces zeros (because numerator is zero) without NaN.
    assert torch.all(torch.isfinite(normalized[1]))


def test_quat_identity_shapes_correctly_for_both_conventions():
    q_xyzw = quat_identity([2, 3], w_last=True)
    assert q_xyzw.shape == (2, 3, 4)
    assert torch.equal(q_xyzw[..., 3], torch.ones(2, 3))
    assert torch.equal(q_xyzw[..., :3], torch.zeros(2, 3, 3))

    q_wxyz = quat_identity([4], w_last=False)
    assert q_wxyz.shape == (4, 4)
    assert torch.equal(q_wxyz[..., 0], torch.ones(4))
    assert torch.equal(q_wxyz[..., 1:], torch.zeros(4, 3))


def test_quat_identity_like_matches_dtype_device_and_leading_shape():
    sample = torch.randn(2, 3, 4, dtype=torch.float64)
    identity = quat_identity_like(sample, w_last=True)
    assert identity.shape == sample.shape
    assert identity.dtype == sample.dtype
    assert torch.equal(identity[..., 3], torch.ones(2, 3, dtype=torch.float64))


def test_quat_mul_identity_is_left_and_right_neutral():
    q = quat_normalize(torch.tensor([[0.1, -0.3, 0.7, 0.6]]))
    eye = quat_identity([1], w_last=True)
    assert torch.allclose(quat_mul(q, eye, w_last=True), q, atol=1e-6)
    assert torch.allclose(quat_mul(eye, q, w_last=True), q, atol=1e-6)


def test_quat_mul_pi_about_z_squares_to_pi_rotation():
    half_pi = math.pi / 2
    q_z90 = torch.tensor(
        [[0.0, 0.0, math.sin(half_pi / 2), math.cos(half_pi / 2)]]
    )
    q_z180 = quat_mul(q_z90, q_z90, w_last=True)
    assert torch.allclose(q_z180[..., 2].abs(), torch.tensor([1.0]), atol=1e-6)
    assert torch.allclose(q_z180[..., 3].abs(), torch.tensor([0.0]), atol=1e-6)


def test_quat_conjugate_inverts_the_xyz_components_only_for_w_last():
    q = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expected = torch.tensor([[-1.0, -2.0, -3.0, 4.0]])
    assert torch.equal(quat_conjugate(q, w_last=True), expected)
    expected_wxyz = torch.tensor([[1.0, -2.0, -3.0, -4.0]])
    assert torch.equal(quat_conjugate(q, w_last=False), expected_wxyz)


def test_quat_mul_with_conjugate_returns_identity_for_unit_quat():
    q = quat_normalize(torch.tensor([[0.2, 0.4, 0.1, 0.7]]))
    product = quat_mul(q, quat_conjugate(q, w_last=True), w_last=True)
    expected = quat_identity([1], w_last=True)
    assert torch.allclose(product, expected, atol=1e-6)


def test_quat_apply_z90_rotation_takes_x_axis_to_y_axis():
    half_pi = math.pi / 2
    q_z90 = torch.tensor(
        [[0.0, 0.0, math.sin(half_pi / 2), math.cos(half_pi / 2)]]
    )
    x_hat = torch.tensor([[1.0, 0.0, 0.0]])
    rotated = quat_apply(q_z90, x_hat, w_last=True)
    assert torch.allclose(rotated, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)


def test_quat_apply_yaw_respects_w_first_quaternion_layout():
    half_pi = math.pi / 2
    q_z90_xyzw = torch.tensor(
        [[0.0, 0.0, math.sin(half_pi / 2), math.cos(half_pi / 2)]]
    )
    q_z90_wxyz = xyzw_to_wxyz(q_z90_xyzw)
    x_hat = torch.tensor([[1.0, 0.0, 0.0]])

    rotated = quat_apply_yaw(q_z90_wxyz, x_hat, w_last=False)

    assert torch.allclose(rotated, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)


def test_quat_rotate_matches_quat_apply_on_arbitrary_unit_quat():
    q = quat_normalize(torch.tensor([[0.1, -0.2, 0.3, 0.5]]))
    v = torch.tensor([[1.0, 2.0, 3.0]])
    apply_result = quat_apply(q, v, w_last=True)
    rotate_result = quat_rotate(q, v, w_last=True)
    assert torch.allclose(apply_result, rotate_result, atol=1e-5)


def test_quat_rotate_inverse_undoes_quat_rotate():
    q = quat_normalize(torch.tensor([[0.5, 0.5, 0.5, 0.5]]))
    v = torch.tensor([[1.0, 2.0, 3.0]])
    rotated = quat_rotate(q, v, w_last=True)
    restored = quat_rotate_inverse(q, rotated, w_last=True)
    assert torch.allclose(restored, v, atol=1e-5)


def test_quat_from_angle_axis_then_angle_axis_round_trips_within_pi():
    angle = torch.tensor([math.pi / 3])
    axis = torch.tensor([[0.0, 0.0, 1.0]])
    q = quat_from_angle_axis(angle, axis, w_last=True)
    recovered_angle, recovered_axis = quat_angle_axis(q, w_last=True)
    assert torch.allclose(recovered_angle, angle, atol=1e-5)
    assert torch.allclose(recovered_axis, axis, atol=1e-5)


def test_quat_pos_flips_negative_real_quaternion_to_positive_real():
    q_neg = torch.tensor([[1.0, 2.0, 3.0, -4.0]])
    q_pos = torch.tensor([[0.5, 0.6, 0.7, 0.0]])
    flipped = quat_pos(q_neg, w_last=True)
    unchanged = quat_pos(q_pos, w_last=True)
    assert torch.equal(flipped, torch.tensor([[-1.0, -2.0, -3.0, 4.0]]))
    assert torch.equal(unchanged, q_pos)


def test_quat_normalize_returns_positive_real_unit_quaternion():
    q = torch.tensor([[2.0, 4.0, 6.0, -8.0]])
    normalized = quat_normalize(q)
    assert torch.allclose(normalized.norm(dim=-1), torch.tensor([1.0]), atol=1e-6)
    assert (normalized[..., 3] >= 0).all()


def test_quat_unit_matches_normalize_for_arbitrary_input():
    q = torch.tensor([[1.5, -0.5, 2.0, 3.0]])
    assert torch.allclose(quat_unit(q), normalize(q), atol=1e-7)


def test_quat_axis_extracts_columns_of_implicit_rotation_matrix():
    q_identity = quat_identity([1], w_last=True)
    x_axis = quat_axis(q_identity, axis=0, w_last=True)
    y_axis = quat_axis(q_identity, axis=1, w_last=True)
    z_axis = quat_axis(q_identity, axis=2, w_last=True)
    assert torch.allclose(x_axis, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6)
    assert torch.allclose(y_axis, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)
    assert torch.allclose(z_axis, torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-6)


def test_quat_from_two_vectors_preserves_float64_dtype_for_same_direction():
    v1 = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    v2 = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)

    quat = quat_from_two_vectors(v1, v2, w_last=True)

    assert quat.dtype == torch.float64
    assert torch.allclose(
        quat,
        torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64),
        atol=1e-12,
    )


def test_quat_from_two_vectors_opposite_non_x_axis_rotates_by_pi():
    v1 = torch.tensor([[0.0, 1.0, 0.0]])
    v2 = torch.tensor([[0.0, -1.0, 0.0]])

    quat = quat_from_two_vectors(v1, v2, w_last=True)
    rotated = quat_apply(quat, v1, w_last=True)

    assert torch.allclose(rotated, v2, atol=1e-6)


def test_quat_from_two_vectors_opposite_non_x_axis_w_first_rotates_by_pi():
    v1 = torch.tensor([[0.0, 1.0, 0.0]])
    v2 = torch.tensor([[0.0, -1.0, 0.0]])

    quat = quat_from_two_vectors(v1, v2, w_last=False)
    rotated = quat_apply(quat, v1, w_last=False)

    assert torch.allclose(rotated, v2, atol=1e-6)
