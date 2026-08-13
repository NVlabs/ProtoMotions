# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for robot anatomical heading helpers."""

from types import SimpleNamespace

import torch


def test_semantic_forward_axis_normalizes_and_supports_legacy_configs():
    from protomotions.utils.robot_heading import (
        robot_semantic_forward_axis_tensor,
        semantic_forward_axis_tensor,
    )

    axis = semantic_forward_axis_tensor((3.0, 4.0))
    torch.testing.assert_close(axis, torch.tensor([0.6, 0.8]))

    legacy_axis = robot_semantic_forward_axis_tensor(SimpleNamespace())
    torch.testing.assert_close(legacy_axis, torch.tensor([1.0, 0.0]))


def test_semantic_forward_xy_uses_the_declared_robot_axis():
    from protomotions.utils.robot_heading import semantic_forward_xy
    from protomotions.utils.rotations import heading_to_quat

    heading = torch.tensor([torch.pi / 2.0])
    root_rotation = heading_to_quat(heading, w_last=True)

    world_forward = semantic_forward_xy(
        root_rotation,
        (0.0, -1.0),
        w_last=True,
    )

    torch.testing.assert_close(
        world_forward,
        torch.tensor([[1.0, 0.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_heading_to_quat_aims_the_declared_axis_at_the_target_heading():
    from protomotions.utils.robot_heading import (
        heading_to_quat_for_forward_axis,
        semantic_forward_xy,
    )

    target_heading = torch.tensor([0.0, torch.pi / 2.0])
    rotation = heading_to_quat_for_forward_axis(
        target_heading,
        (0.0, -1.0),
        w_last=True,
    )

    world_forward = semantic_forward_xy(
        rotation,
        (0.0, -1.0),
        w_last=True,
    )
    expected = torch.stack(
        (torch.cos(target_heading), torch.sin(target_heading)), dim=-1
    )
    torch.testing.assert_close(world_forward, expected, atol=1.0e-6, rtol=0.0)
