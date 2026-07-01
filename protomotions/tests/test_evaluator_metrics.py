# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for evaluator motion metric storage and reductions."""

import pytest
import torch

from protomotions.agents.evaluators.metrics import MotionMetrics


def test_motion_metrics_update_masks_invalid_frames_and_reduces_scalar_features():
    metrics = MotionMetrics(
        num_motions=3,
        motion_lens=torch.tensor([2, 3, 1]),
        max_motion_len=3,
        num_sub_features=1,
        device=torch.device("cpu"),
    )

    metrics.update(torch.tensor([0, 1]), torch.tensor([1.0, 10.0]))
    metrics.update(
        torch.tensor([0, 1, 2]),
        torch.tensor([3.0, 20.0, 99.0]),
        frame_indices=torch.tensor([1, 2, 2]),
    )

    assert torch.equal(metrics.frame_counts, torch.tensor([2, 2, 0]))
    assert torch.equal(metrics.data[0, :, 0], torch.tensor([1.0, 3.0, 0.0]))
    assert torch.equal(metrics.data[1, :, 0], torch.tensor([10.0, 0.0, 20.0]))
    assert torch.equal(metrics.data[2, :, 0], torch.zeros(3))
    assert torch.equal(
        metrics.get_unfilled_mask()[:, :, 0],
        torch.tensor(
            [
                [False, False, True],
                [False, False, True],
                [True, True, True],
            ]
        ),
    )

    max_values, max_frames = metrics.max_reduce_each_motion(with_frame=True)
    assert torch.equal(max_values, torch.tensor([3.0, 10.0, float("-inf")]))
    assert torch.equal(max_frames[:, 0], torch.tensor([1, 0, 0]))
    assert torch.equal(
        metrics.min_reduce_each_motion(),
        torch.tensor([1.0, 0.0, float("inf")]),
    )
    assert torch.equal(
        metrics.mean_reduce_each_motion(),
        torch.tensor([2.0, 5.0, 0.0]),
    )
    assert metrics.max_mean_reduce() == pytest.approx(6.5)
    assert metrics.min_mean_reduce() == pytest.approx(0.5)
    assert metrics.mean_mean_reduce() == pytest.approx(3.5)


def test_motion_metrics_validates_update_shapes_and_unique_motion_ids():
    metrics = MotionMetrics(
        num_motions=2,
        motion_lens=torch.tensor([2, 2]),
        max_motion_len=2,
        device=torch.device("cpu"),
    )

    with pytest.raises(AssertionError):
        metrics.update(torch.tensor([0, 0]), torch.tensor([1.0, 2.0]))

    with pytest.raises(AssertionError):
        metrics.update(
            torch.tensor([0, 1]),
            torch.tensor([1.0, 2.0]),
            frame_indices=torch.tensor([0]),
        )


def test_motion_metrics_vector_reductions_and_external_ops():
    metrics = MotionMetrics(
        num_motions=3,
        motion_lens=torch.tensor([2, 2, 2]),
        max_motion_len=2,
        num_sub_features=2,
        device=torch.device("cpu"),
    )
    metrics.data[0] = torch.tensor([[1.0, 4.0], [3.0, 2.0]])
    metrics.data[1] = torch.tensor([[5.0, 0.0], [1.0, 8.0]])
    metrics.frame_counts[:] = torch.tensor([2, 1, 0])

    def value_range(metric):
        return metric.max_reduce_each_motion() - metric.min_reduce_each_motion()

    assert torch.equal(
        metrics.mean_reduce_each_motion(),
        torch.tensor([[2.0, 3.0], [5.0, 0.0], [0.0, 0.0]]),
    )
    assert torch.equal(metrics.mean_max_reduce(), torch.tensor([5.0, 3.0]))
    assert torch.equal(metrics.mean_min_reduce(), torch.tensor([2.0, 0.0]))
    assert torch.equal(metrics.ops_mean_reduce(value_range), torch.tensor([1.0, 1.0]))

    empty = MotionMetrics(
        num_motions=2,
        motion_lens=torch.tensor([2, 2]),
        max_motion_len=2,
        num_sub_features=2,
        device=torch.device("cpu"),
    )
    assert torch.equal(empty.mean_max_reduce(), torch.zeros(2))
    assert torch.equal(empty.mean_min_reduce(), torch.zeros(2))

    empty_scalar = MotionMetrics(
        num_motions=1,
        motion_lens=torch.tensor([2]),
        max_motion_len=2,
        num_sub_features=1,
        device=torch.device("cpu"),
    )
    assert empty_scalar.mean_max_reduce() == pytest.approx(0.0)
    assert empty_scalar.mean_min_reduce() == pytest.approx(0.0)


def test_motion_metrics_finite_difference_jitter_and_validation():
    metrics = MotionMetrics(
        num_motions=2,
        motion_lens=torch.tensor([4, 2]),
        max_motion_len=4,
        num_sub_features=3,
        device=torch.device("cpu"),
    )
    metrics.data[0] = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ]
    )
    metrics.data[1] = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [99.0, 99.0, 99.0],
            [99.0, 99.0, 99.0],
        ]
    )
    metrics.frame_counts[:] = torch.tensor([4, 2])

    assert torch.equal(
        metrics.compute_jitter_reduce_each_motion(num_bodies=1, aggregate_method="sum"),
        torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]),
    )
    assert torch.equal(
        metrics.compute_rotation_jitter_reduce_each_motion(num_bodies=1),
        torch.tensor([[0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 0.0, 0.0]]),
    )
    assert torch.equal(
        metrics.jitter_mean_reduce_each_motion(num_bodies=1, aggregate_method="sum"),
        torch.tensor([0.5, 0.0]),
    )
    assert torch.equal(
        metrics.rotation_jitter_mean_reduce_each_motion(num_bodies=1),
        torch.tensor([1.5, 1.0]),
    )

    multi_body = MotionMetrics(
        num_motions=1,
        motion_lens=torch.tensor([3]),
        max_motion_len=3,
        num_sub_features=6,
        device=torch.device("cpu"),
    )
    multi_body.data[0] = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 4.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 5.0, 0.0, 0.0],
        ]
    )
    multi_body.frame_counts[:] = 3
    assert torch.equal(
        multi_body.compute_rotation_jitter_reduce_each_motion(
            num_bodies=2,
            aggregate_method="max",
        ),
        torch.tensor([[0.0, 4.0, 2.0]]),
    )

    with pytest.raises(AssertionError, match="Expected num_sub_features"):
        metrics.compute_jitter_reduce_each_motion(num_bodies=2)
    with pytest.raises(AssertionError, match="Only 1st and 2nd order"):
        metrics.compute_finite_difference_jitter_reduce_each_motion(
            num_bodies=1,
            order=3,
        )
    with pytest.raises(ValueError, match="Unknown aggregate_method"):
        metrics.compute_jitter_reduce_each_motion(
            num_bodies=1,
            aggregate_method="median",
        )

    too_short = MotionMetrics(
        num_motions=1,
        motion_lens=torch.tensor([1]),
        max_motion_len=1,
        num_sub_features=3,
        device=torch.device("cpu"),
    )
    too_short.frame_counts[:] = 1
    assert torch.equal(
        too_short.compute_rotation_jitter_reduce_each_motion(num_bodies=1),
        torch.zeros(1, 1),
    )


def test_motion_metrics_copy_merge_reset_and_to_device():
    source = MotionMetrics(
        num_motions=2,
        motion_lens=torch.tensor([2, 2]),
        max_motion_len=2,
        num_sub_features=1,
        device=torch.device("cpu"),
    )
    source.data[:, :, 0] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    source.frame_counts[:] = torch.tensor([2, 1])
    target = MotionMetrics(
        num_motions=2,
        motion_lens=torch.tensor([2, 2]),
        max_motion_len=2,
        num_sub_features=1,
        device=torch.device("cpu"),
    )

    target.copy_from(source)
    source.data.zero_()
    assert torch.equal(target.data[:, :, 0], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.equal(target.frame_counts, torch.tensor([2, 1]))

    partial = MotionMetrics(
        num_motions=2,
        motion_lens=torch.tensor([2, 2]),
        max_motion_len=2,
        num_sub_features=1,
        device=torch.device("cpu"),
    )
    partial.copy_from_motion_ids(target, torch.tensor([1]))
    assert torch.equal(partial.data[1], target.data[1])
    assert partial.frame_counts[1] == 1

    extra = MotionMetrics(
        num_motions=1,
        motion_lens=torch.tensor([1]),
        max_motion_len=2,
        num_sub_features=1,
        device=torch.device("cpu"),
    )
    extra.frame_counts[:] = 1
    target.merge_from(extra)
    assert target.num_motions == 3
    assert torch.equal(target.motion_lens, torch.tensor([2, 2, 1]))
    assert target.to(torch.device("cpu")) is target
    assert target.device == torch.device("cpu")

    mismatch = MotionMetrics(
        num_motions=1,
        motion_lens=torch.tensor([1]),
        max_motion_len=3,
        num_sub_features=1,
        device=torch.device("cpu"),
    )
    with pytest.raises(AssertionError):
        target.merge_from(mismatch)

    target.reset()
    assert torch.count_nonzero(target.data) == 0
    assert torch.count_nonzero(target.frame_counts) == 0
