# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for small shared utility modules."""

import numpy as np
import pytest
import torch
import trimesh

from protomotions.utils import mesh_utils, motion_interpolation_utils
from protomotions.utils import torch_utils


def test_mesh_utils_converts_scenes_and_computes_bounding_boxes():
    first = trimesh.Trimesh(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.0, 1.0, 0.0]]),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )
    second = trimesh.Trimesh(
        vertices=np.array([[2.0, -1.0, 1.0], [4.0, 1.0, 2.0], [2.0, 0.0, 5.0]]),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )
    scene = trimesh.Scene()
    scene.add_geometry(first)
    scene.add_geometry(second)

    combined = mesh_utils.as_mesh(scene)
    same_mesh = mesh_utils.as_mesh(first)
    bbox = mesh_utils.compute_bounding_box(combined)

    assert same_mesh is first
    assert combined.vertices.shape == (6, 3)
    assert bbox == pytest.approx((4.0, 3.0, 5.0, 0.0, -1.0, 0.0))


def test_motion_interpolation_handles_positions_quats_and_frame_blends():
    pos0 = torch.tensor([[0.0, 0.0], [10.0, 20.0]])
    pos1 = torch.tensor([[2.0, 4.0], [20.0, 40.0]])
    blend = torch.tensor([0.25, 0.5])
    assert torch.allclose(
        motion_interpolation_utils.interpolate_pos(pos0, pos1, blend),
        torch.tensor([[0.5, 1.0], [15.0, 30.0]]),
    )

    pos0_3d = torch.zeros(2, 2, 3)
    pos1_3d = torch.ones(2, 2, 3)
    assert torch.allclose(
        motion_interpolation_utils.interpolate_pos(pos0_3d, pos1_3d, blend),
        torch.tensor(
            [
                [[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]],
                [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
            ]
        ),
    )

    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
    assert torch.allclose(
        motion_interpolation_utils.interpolate_quat(quat, quat, blend),
        quat,
    )
    quat_3d = quat[:, None, :].expand(-1, 2, -1)
    assert torch.allclose(
        motion_interpolation_utils.interpolate_quat(quat_3d, quat_3d, blend),
        quat_3d,
    )

    frame0, frame1, frame_blend = motion_interpolation_utils.calc_frame_blend(
        time=torch.tensor([0.0, 0.5, 2.0]),
        length=torch.tensor([1.0, 1.0, 1.0]),
        num_frames=torch.tensor([5, 5, 5]),
        dt=torch.tensor([0.25, 0.25, 0.25]),
    )
    assert torch.equal(frame0, torch.tensor([0, 2, 4]))
    assert torch.equal(frame1, torch.tensor([1, 3, 4]))
    assert torch.allclose(frame_blend, torch.tensor([0.0, 0.0, 0.0]))


def test_motion_interpolation_rejects_unsupported_tensor_rank():
    with pytest.raises(ValueError, match="pos1 has 4 dimensions"):
        motion_interpolation_utils.interpolate_pos(
            torch.zeros(1, 1, 1, 3),
            torch.zeros(1, 1, 1, 3),
            torch.zeros(1),
        )

    with pytest.raises(ValueError, match="rot1 has 4 dimensions"):
        motion_interpolation_utils.interpolate_quat(
            torch.zeros(1, 1, 1, 4),
            torch.zeros(1, 1, 1, 4),
            torch.zeros(1),
        )


def test_torch_utils_grad_norm_to_torch_and_seeding(monkeypatch, capsys):
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter.grad = torch.tensor([3.0, 4.0])
    no_grad_parameter = torch.nn.Parameter(torch.tensor([5.0]))

    assert torch.allclose(
        torch_utils.grad_norm([parameter, no_grad_parameter]),
        torch.tensor(5.0),
    )
    converted = torch_utils.to_torch(
        [1, 2],
        dtype=torch.int64,
        device="cpu",
        requires_grad=False,
    )
    assert torch.equal(converted, torch.tensor([1, 2], dtype=torch.int64))

    cuda_seed_calls = []
    deterministic_calls = []
    monkeypatch.setattr(
        torch_utils.torch.cuda,
        "manual_seed",
        lambda seed: cuda_seed_calls.append(("manual_seed", seed)),
    )
    monkeypatch.setattr(
        torch_utils.torch.cuda,
        "manual_seed_all",
        lambda seed: cuda_seed_calls.append(("manual_seed_all", seed)),
    )
    monkeypatch.setattr(
        torch_utils.torch,
        "use_deterministic_algorithms",
        lambda enabled: deterministic_calls.append(enabled),
    )

    assert torch_utils.seeding(123, torch_deterministic=False) == 123
    assert torch_utils.seeding(456, torch_deterministic=True) == 456

    assert "Setting seed: 123" in capsys.readouterr().out
    assert ("manual_seed", 123) in cuda_seed_calls
    assert ("manual_seed_all", 456) in cuda_seed_calls
    assert deterministic_calls == [True]
    assert torch_utils.os.environ["PYTHONHASHSEED"] == "456"
    assert torch_utils.os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
