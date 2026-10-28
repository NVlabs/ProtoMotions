# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""G16 tracking kernels: reference regimes, reductions and coordinate alignment."""

import pytest
import torch

from protomotions.envs.terminations.foot_tracking import (
    aligned_foot_position_termination,
)
from protomotions.envs.terminations.low_pose import (
    tracking_failure_ratio,
    upright_foot_height_failed,
    upright_foot_position_failed,
)


def ratio(current, height, **kwargs):
    return tracking_failure_ratio(
        current,
        torch.zeros_like(current),
        height,
        upper_body_indices=[0, 1, 2],
        foot_indices=[3, 4],
        upright_limit=0.5,
        **kwargs,
    )


@pytest.mark.parametrize("mean", [False, True])
@pytest.mark.parametrize("subset", [None, [0, 1, 2, 3, 4]])
def test_upright_matches_original_reduction(mean, subset):
    current = torch.randn(100, 8, 3, generator=torch.Generator().manual_seed(7))
    selected = current if subset is None else current[:, subset]
    distances = selected.norm(dim=-1)
    expected = (distances.mean(-1) if mean else distances.amax(-1)) / 0.5
    torch.testing.assert_close(
        ratio(
            current,
            torch.full((100,), 0.5),
            upright_body_indices=subset,
            upright_mean=mean,
        ),
        expected,
    )


def test_low_pose_ignores_foot_xy_and_other_bodies():
    current = torch.zeros(1, 8, 3)
    current[:, 3:5, :2] = 100
    current[:, 5:] = 100
    assert ratio(current, torch.tensor([0.49])).item() == 0


def test_low_pose_strict_boundaries():
    current = torch.zeros(4, 8, 3, dtype=torch.float64)
    current[0, 0, 0] = 0.5
    current[1, 1, 0] = 0.501
    current[2, 3, 2] = 0.3
    current[3, 4, 2] = 0.301
    assert (ratio(current, torch.full((4,), 0.49)) > 1).tolist() == [
        False,
        True,
        False,
        True,
    ]


def test_reference_height_selects_regime_per_environment():
    current = torch.zeros(2, 8, 3)
    current[:, 3, 0] = 0.6
    assert (ratio(current, torch.tensor([0.49, 0.5])) > 1).tolist() == [False, True]


def test_upright_mean_is_not_training_max():
    current = torch.zeros(1, 8, 3)
    current[:, 0, 0] = 0.6
    height = torch.ones(1)
    assert (ratio(current, height) > 1).item()
    assert not (ratio(current, height, upright_mean=True) > 1).item()


def test_empty_batch_and_inputs_are_unchanged():
    current = torch.zeros(0, 8, 3)
    assert ratio(current, torch.empty(0)).shape == (0,)
    current = torch.randn(3, 8, 3)
    before = current.clone()
    ratio(current, torch.tensor([0.3, 0.5, 1.0]))
    torch.testing.assert_close(current, before)


def poses(batch):
    position = torch.zeros(batch, 5, 3, dtype=torch.float64)
    rotation = torch.zeros(batch, 5, 4, dtype=torch.float64)
    rotation[..., 3] = 1
    return position, rotation


@pytest.mark.parametrize("foot", [3, 4])
def test_aligned_xyz_strict_limit_and_other_bodies(foot):
    reference, rotation = poses(3)
    current = reference.clone()
    current[:, foot, 0] = torch.tensor([0.2, 0.20001, 0], dtype=current.dtype)
    current[2, 1] = 100
    assert aligned_foot_position_termination(
        current, reference, rotation, rotation, [3, 4], 0
    ).tolist() == [False, True, False]


def test_alignment_matches_xy_and_yaw_but_retains_world_height():
    reference, ref_rotation = poses(1)
    reference[0, 0, 2] = 1
    reference[0, 3, 0] = 1
    reference[0, 4, 0] = -1
    current = reference.clone()
    current[..., 0] = 4
    current[..., 1] = 5
    current[0, 3, 1] += 1
    current[0, 4, 1] -= 1
    rotation = ref_rotation.clone()
    rotation[..., 2:] = 2**-0.5
    before = current.clone()
    assert not aligned_foot_position_termination(
        current, reference, rotation, ref_rotation, [3, 4], 0
    ).item()
    torch.testing.assert_close(current, before)
    current[..., 2] += 0.21
    assert aligned_foot_position_termination(
        current, reference, rotation, ref_rotation, [3, 4], 0
    ).item()


def test_upright_foot_checks_disable_only_in_low_regime():
    reference, rotation = poses(4)
    current = reference.clone()
    current[:, 3, 2] = torch.tensor([0.15, 0.15001, 0.3, 0.3], dtype=current.dtype)
    heights = torch.tensor([1.0, 1.0, 0.5, 0.49])
    assert upright_foot_height_failed(current, reference, heights, [3, 4]).tolist() == [
        False,
        True,
        True,
        False,
    ]
    assert upright_foot_position_failed(
        current, reference, rotation, rotation, heights, [3, 4], 0
    ).tolist() == [False, False, True, False]


def test_foot_checks_support_empty_batches():
    position, rotation = poses(0)
    assert aligned_foot_position_termination(
        position, position, rotation, rotation, [3, 4], 0
    ).shape == (0,)
    assert upright_foot_height_failed(
        position, position, torch.empty(0), [3, 4]
    ).shape == (0,)
