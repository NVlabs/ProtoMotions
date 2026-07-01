# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-executed parity tests for protomotions.utils.rotations.

Most rotation helpers are TorchScript exports. These tests load a second copy of
the module with scripting disabled so coverage sees Python source execution,
then assert parity with the real scripted exports and known numerical behavior.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from protomotions.utils import rotations as scripted_rotations


@pytest.fixture(scope="module")
def source_rotations():
    rotations_path = (
        Path(__file__).resolve().parents[1] / "utils" / "rotations.py"
    )
    module_name = "_rotations_source_for_coverage"
    spec = importlib.util.spec_from_file_location(module_name, rotations_path)
    module = importlib.util.module_from_spec(spec)
    original_script = torch.jit.script
    torch.jit.script = lambda fn=None, *args, **kwargs: fn
    try:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        torch.jit.script = original_script
        sys.modules.pop(module_name, None)
    return module


def _assert_same_rotation(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual.abs(), expected.abs(), atol=1e-6, rtol=1e-6)


def _z_quat(angle: float, w_last: bool, dtype=torch.float64) -> torch.Tensor:
    half = angle / 2
    if w_last:
        return torch.tensor(
            [[0.0, 0.0, torch.sin(torch.tensor(half)), torch.cos(torch.tensor(half))]],
            dtype=dtype,
        )
    return torch.tensor(
        [[torch.cos(torch.tensor(half)), 0.0, 0.0, torch.sin(torch.tensor(half))]],
        dtype=dtype,
    )


def _assert_parity(source_value, scripted_value) -> None:
    if isinstance(source_value, tuple):
        assert isinstance(scripted_value, tuple)
        assert len(source_value) == len(scripted_value)
        for source_item, scripted_item in zip(source_value, scripted_value):
            torch.testing.assert_close(source_item, scripted_item, atol=1e-6, rtol=1e-6)
        return
    torch.testing.assert_close(source_value, scripted_value, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("w_last", [True, False])
def test_basic_quaternion_helpers_match_scripted_exports_and_known_ops(
    source_rotations, w_last
):
    q_z90 = _z_quat(torch.pi / 2, w_last)
    q_z180 = _z_quat(torch.pi, w_last)
    identity = (
        torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        if w_last
        else torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    )
    x_hat = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    y_hat = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    raw_q = 2.0 * q_z90

    for name, args in [
        ("quat_mul", (q_z90, q_z90, w_last)),
        ("quat_conjugate", (q_z90, w_last)),
        ("quat_apply", (q_z90, x_hat, w_last)),
        ("quat_rotate", (q_z90, x_hat, w_last)),
        ("quat_rotate_inverse", (q_z90, y_hat, w_last)),
        ("quat_unit", (raw_q,)),
        ("quat_pos", (-q_z90, w_last)),
        ("quat_mul_norm", (q_z90, q_z90, w_last)),
    ]:
        _assert_parity(
            getattr(source_rotations, name)(*args),
            getattr(scripted_rotations, name)(*args),
        )

    normalized = source_rotations.normalize(torch.tensor([[3.0, 4.0]], dtype=torch.float64))
    _assert_parity(
        normalized,
        scripted_rotations.normalize(torch.tensor([[3.0, 4.0]], dtype=torch.float64)),
    )
    torch.testing.assert_close(
        normalized,
        torch.tensor([[0.6, 0.8]], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )

    source_identity = source_rotations.quat_identity([2], w_last)
    _assert_parity(source_identity, scripted_rotations.quat_identity([2], w_last))
    torch.testing.assert_close(
        source_identity,
        identity.expand(2, 4),
        atol=1e-12,
        rtol=1e-12,
    )
    like = torch.empty(2, 4, dtype=torch.float64)
    source_identity_like = source_rotations.quat_identity_like(like, w_last)
    scripted_identity_like = scripted_rotations.quat_identity_like(like, w_last)
    _assert_parity(source_identity_like, scripted_identity_like)
    assert source_identity_like.dtype == torch.float64
    assert scripted_identity_like.dtype == torch.float64
    torch.testing.assert_close(scripted_identity_like, identity.to(torch.float64).expand(2, 4))

    xyzw = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.float64)
    wxyz = torch.tensor([[[4.0, 1.0, 2.0, 3.0]]], dtype=torch.float64)
    _assert_parity(source_rotations.xyzw_to_wxyz(xyzw), scripted_rotations.xyzw_to_wxyz(xyzw))
    _assert_parity(source_rotations.wxyz_to_xyzw(wxyz), scripted_rotations.wxyz_to_xyzw(wxyz))
    torch.testing.assert_close(source_rotations.xyzw_to_wxyz(xyzw), wxyz)
    torch.testing.assert_close(source_rotations.wxyz_to_xyzw(wxyz), xyzw)

    _assert_same_rotation(source_rotations.quat_mul(q_z90, q_z90, w_last), q_z180)
    torch.testing.assert_close(source_rotations.quat_apply(q_z90, x_hat, w_last), y_hat)
    torch.testing.assert_close(source_rotations.quat_rotate(q_z90, x_hat, w_last), y_hat)
    torch.testing.assert_close(source_rotations.quat_rotate_inverse(q_z90, y_hat, w_last), x_hat)
    _assert_same_rotation(source_rotations.quat_mul_norm(q_z90, q_z90, w_last), q_z180)


@pytest.mark.parametrize("w_last", [True, False])
def test_heading_helpers_match_scripted_exports_and_project_to_xy_plane(
    source_rotations, w_last
):
    headings = torch.tensor([0.0, torch.pi / 2, -torch.pi / 4])
    heading_vecs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [2.0**-0.5, -(2.0**-0.5)]],
    )
    axis = torch.zeros(headings.shape + (3,))
    axis[..., 2] = 1
    q_heading = scripted_rotations.quat_from_angle_axis(headings, axis, w_last=w_last)

    source_heading_q = source_rotations.heading_to_quat(headings, w_last=w_last)
    _assert_parity(
        source_heading_q,
        scripted_rotations.heading_to_quat(headings, w_last=w_last),
    )
    torch.testing.assert_close(
        source_heading_q,
        q_heading,
        atol=1e-6,
        rtol=1e-6,
    )
    x_hat = torch.tensor([[1.0, 0.0, 0.0]]).expand(3, 3)

    for name, args in [
        ("vec_to_heading", (heading_vecs,)),
        ("calc_heading", (q_heading, w_last)),
        ("calc_heading_quat", (q_heading, w_last)),
        ("calc_heading_quat_inv", (q_heading, w_last)),
        ("quat_apply_yaw", (q_heading, x_hat, w_last)),
        ("quat_axis", (q_heading, 0, w_last)),
        ("get_basis_vector", (q_heading, x_hat, w_last)),
        ("normalize_angle", (torch.tensor([3 * torch.pi, -3 * torch.pi / 2]),)),
    ]:
        _assert_parity(
            getattr(source_rotations, name)(*args),
            getattr(scripted_rotations, name)(*args),
        )

    torch.testing.assert_close(source_rotations.vec_to_heading(heading_vecs), headings)
    torch.testing.assert_close(
        source_rotations.calc_heading(q_heading, w_last),
        headings,
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        source_rotations.quat_apply_yaw(q_heading, x_hat, w_last)[..., :2],
        heading_vecs,
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("w_last", [True, False])
def test_euler_angle_and_matrix_helpers_match_scripted_exports(source_rotations, w_last):
    roll = torch.tensor([0.0, torch.pi / 2], dtype=torch.float64)
    pitch = torch.tensor([0.0, 0.0], dtype=torch.float64)
    yaw = torch.tensor([torch.pi / 3, 0.0], dtype=torch.float64)
    q = scripted_rotations.quat_from_euler_xyz(roll, pitch, yaw, w_last=w_last)
    z_mats = scripted_rotations.quaternion_to_matrix(
        scripted_rotations.heading_to_quat(
            torch.tensor([torch.pi / 6, -torch.pi / 4], dtype=torch.float64),
            w_last=w_last,
        ),
        w_last=w_last,
    )

    for name, args in [
        ("quat_from_euler_xyz", (roll, pitch, yaw, w_last)),
        ("get_euler_xyz", (q, w_last)),
        ("quat_diff_rad", (q, q, w_last)),
        ("quat_diff_norm", (q, q, w_last)),
        ("quat_angle_diff_norm", (q, q, w_last)),
        ("quaternion_to_matrix", (q, w_last)),
        ("matrix_to_quaternion", (scripted_rotations.quaternion_to_matrix(q, w_last), w_last)),
        ("angle_from_matrix_axis", (z_mats, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))),
    ]:
        _assert_parity(
            getattr(source_rotations, name)(*args),
            getattr(scripted_rotations, name)(*args),
        )

    source_roll, source_pitch, source_yaw = source_rotations.get_euler_xyz(q, w_last)
    torch.testing.assert_close(source_roll, roll, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(source_pitch, pitch, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(source_yaw, yaw, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        source_rotations.quat_diff_rad(q, q, w_last),
        torch.zeros(2, dtype=torch.float64),
    )
    torch.testing.assert_close(
        source_rotations.angle_from_matrix_axis(
            z_mats, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        ),
        torch.tensor([torch.pi / 6, -torch.pi / 4], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize("w_last", [True, False])
def test_exp_map_and_tan_norm_helpers_match_scripted_exports_and_round_trip(
    source_rotations, w_last
):
    exp_map = torch.tensor(
        [[0.0, 0.0, torch.pi / 4], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    angle = torch.tensor([torch.pi / 4, 0.0], dtype=torch.float64)
    axis = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    q = scripted_rotations.exp_map_to_quat(exp_map, w_last=w_last)
    tan_norm = scripted_rotations.quat_to_tan_norm(q, w_last=w_last)

    for name, args in [
        ("angle_axis_to_exp_map", (angle, axis)),
        ("exp_map_to_angle_axis", (exp_map,)),
        ("exp_map_to_quat", (exp_map, w_last)),
        ("quat_to_exp_map", (q, w_last)),
        ("quat_to_tan_norm", (q, w_last)),
        ("tan_norm_to_quat", (tan_norm, w_last)),
    ]:
        _assert_parity(
            getattr(source_rotations, name)(*args),
            getattr(scripted_rotations, name)(*args),
        )

    torch.testing.assert_close(source_rotations.angle_axis_to_exp_map(angle, axis), exp_map)
    recovered_angle, recovered_axis = source_rotations.exp_map_to_angle_axis(exp_map)
    torch.testing.assert_close(recovered_angle, angle, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(recovered_axis, axis, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(source_rotations.quat_to_exp_map(q, w_last), exp_map)
    torch.testing.assert_close(tan_norm[..., :3].norm(dim=-1), torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(tan_norm[..., 3:].norm(dim=-1), torch.ones(2, dtype=torch.float64))
    _assert_same_rotation(source_rotations.tan_norm_to_quat(tan_norm, w_last), q)


@pytest.mark.parametrize("w_last", [True, False])
def test_quat_from_two_vectors_matches_scripted_exports_for_special_cases(
    source_rotations, w_last
):
    v1 = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )
    v2 = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
    )

    source = source_rotations.quat_from_two_vectors(v1, v2, w_last=w_last)
    scripted = scripted_rotations.quat_from_two_vectors(v1, v2, w_last=w_last)

    _assert_parity(source, scripted)
    torch.testing.assert_close(
        source_rotations.quat_apply(source, v1, w_last),
        v2,
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("w_last", [True, False])
def test_quat_angle_axis_matches_scripted_exports_for_shortest_angle_and_shapes(
    source_rotations, w_last
):
    axis = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    q = scripted_rotations.quat_from_angle_axis(
        torch.tensor([torch.pi / 2, torch.pi / 3], dtype=torch.float64),
        axis,
        w_last=w_last,
    )
    q[1] *= -1.0
    q = q.reshape(1, 2, 4)

    source_angle, source_axis = source_rotations.quat_angle_axis(q, w_last)
    scripted_angle, scripted_axis = scripted_rotations.quat_angle_axis(q, w_last)

    assert source_angle.shape == (1, 2)
    assert source_axis.shape == (1, 2, 3)
    torch.testing.assert_close(source_angle, scripted_angle, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(source_axis, scripted_axis, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        source_angle,
        torch.tensor([[torch.pi / 2, torch.pi / 3]], dtype=torch.float64),
    )
    assert torch.all(source_angle <= torch.pi)


@pytest.mark.parametrize("w_last", [True, False])
def test_quat_from_two_vectors_branch_guards_match_scripted_exports(
    source_rotations, w_last
):
    same = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    source_same = source_rotations.quat_from_two_vectors(same, same, w_last=w_last)
    scripted_same = scripted_rotations.quat_from_two_vectors(same, same, w_last=w_last)
    _assert_parity(source_same, scripted_same)
    assert source_same.dtype == torch.float64

    opposite_sources = torch.tensor(
        [[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]],
        dtype=torch.float64,
    )
    opposite_targets = -opposite_sources
    source_opposite = source_rotations.quat_from_two_vectors(
        opposite_sources, opposite_targets, w_last=w_last
    )
    scripted_opposite = scripted_rotations.quat_from_two_vectors(
        opposite_sources, opposite_targets, w_last=w_last
    )

    _assert_parity(source_opposite, scripted_opposite)
    torch.testing.assert_close(
        source_rotations.quat_apply(source_opposite, opposite_sources, w_last),
        opposite_targets,
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("w_last", [True, False])
def test_axis_angle_to_quaternion_uses_small_angle_series_and_preserves_shape(
    source_rotations, w_last
):
    axis_angle = torch.tensor(
        [
            [[1.0e-8, 0.0, 0.0], [0.0, 0.0, torch.pi / 2]],
            [[0.0, -2.0e-8, 0.0], [0.0, torch.pi, 0.0]],
        ],
        dtype=torch.float64,
    )

    source = source_rotations.axis_angle_to_quaternion(axis_angle, w_last=w_last)
    scripted = scripted_rotations.axis_angle_to_quaternion(axis_angle, w_last=w_last)

    assert source.shape == (2, 2, 4)
    torch.testing.assert_close(source, scripted, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        source.norm(dim=-1), torch.ones(2, 2, dtype=torch.float64)
    )

    if w_last:
        torch.testing.assert_close(
            source[0, 0],
            torch.tensor([5.0e-9, 0.0, 0.0, 1.0], dtype=torch.float64),
        )
        _assert_same_rotation(
            source[0, 1],
            torch.tensor([0.0, 0.0, 2.0**-0.5, 2.0**-0.5], dtype=torch.float64),
        )
    else:
        torch.testing.assert_close(
            source[0, 0],
            torch.tensor([1.0, 5.0e-9, 0.0, 0.0], dtype=torch.float64),
        )
        _assert_same_rotation(
            source[0, 1],
            torch.tensor([2.0**-0.5, 0.0, 0.0, 2.0**-0.5], dtype=torch.float64),
        )


@pytest.mark.parametrize("w_last", [True, False])
def test_matrix_to_quaternion_covers_all_largest_component_branches(
    source_rotations, w_last
):
    quats_wxyz = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    quats = (
        scripted_rotations.wxyz_to_xyzw(quats_wxyz)
        if w_last
        else quats_wxyz
    )
    matrices = scripted_rotations.quaternion_to_matrix(quats, w_last=w_last)

    source = source_rotations.matrix_to_quaternion(matrices, w_last=w_last)
    scripted = scripted_rotations.matrix_to_quaternion(matrices, w_last=w_last)

    assert source.shape == (4, 4)
    torch.testing.assert_close(source, scripted, atol=1e-12, rtol=1e-12)
    _assert_same_rotation(source, quats)


@pytest.mark.parametrize("w_last", [True, False])
def test_matrix_to_quaternion_rejects_invalid_matrix_shapes_like_scripted_export(
    source_rotations, w_last
):
    invalid_matrix = torch.zeros(2, 2, 3)

    with pytest.raises((RuntimeError, ValueError), match="Invalid rotation matrix shape"):
        source_rotations.matrix_to_quaternion(invalid_matrix, w_last=w_last)
    with pytest.raises(
        (torch.jit.Error, RuntimeError, ValueError),
        match="Invalid rotation matrix shape",
    ):
        scripted_rotations.matrix_to_quaternion(invalid_matrix, w_last=w_last)


@pytest.mark.parametrize("w_last", [True, False])
def test_quat_to_angle_axis_small_angle_fallback_is_finite_and_shape_stable(
    source_rotations, w_last
):
    identity = (
        torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
        if w_last
        else torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    )
    z_quarter = scripted_rotations.axis_angle_to_quaternion(
        torch.tensor([[0.0, 0.0, torch.pi / 2]], dtype=torch.float64),
        w_last=w_last,
    )
    q = torch.cat([identity, z_quarter], dim=0).reshape(1, 2, 4)

    source_angle, source_axis = source_rotations.quat_to_angle_axis(
        q, w_last=w_last
    )
    scripted_angle, scripted_axis = scripted_rotations.quat_to_angle_axis(
        q, w_last=w_last
    )

    assert source_angle.shape == (1, 2)
    assert source_axis.shape == (1, 2, 3)
    torch.testing.assert_close(
        source_angle, scripted_angle, atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        source_axis, scripted_axis, atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        source_angle[0, 0], torch.tensor(0.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        source_axis[0, 0], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        source_angle[0, 1], torch.tensor(torch.pi / 2, dtype=torch.float64)
    )
    torch.testing.assert_close(
        source_axis[0, 1], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    )


def test_slerp_short_path_branches_match_scripted_exports(source_rotations):
    q0 = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    q1 = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0e-5, 0.0, 0.0, (1.0 - 1.0e-10) ** 0.5],
            [0.0, 0.0, -(2.0**-0.5), -(2.0**-0.5)],
        ],
        dtype=torch.float64,
    )
    t = torch.tensor([[0.25], [0.5], [0.5]], dtype=torch.float64)

    source = source_rotations.slerp(q0, q1, t)
    scripted = scripted_rotations.slerp(q0, q1, t)

    assert source.shape == q0.shape
    torch.testing.assert_close(source, scripted, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(source[0], q0[0])
    torch.testing.assert_close(source[1], 0.5 * q0[1] + 0.5 * q1[1])
    _assert_same_rotation(
        source[2],
        torch.tensor(
            [0.0, 0.0, 0.3826834323650898, 0.9238795325112867],
            dtype=torch.float64,
        ),
    )
