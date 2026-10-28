# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference-only recovery grace without episode-local history."""
from types import SimpleNamespace

import pytest
import torch

from protomotions.envs.control.mimic_control import MimicControl, MimicControlConfig
from protomotions.tests.support.fake_reference import fake_reference
from protomotions.tests.support.task_inputs import make_task_inputs


def test_grace_keeps_low_pose_height_for_half_a_second():
    def positions(ids, times):
        result = torch.zeros(len(ids), 1, 3)
        result[:, 0, 2] = torch.where(times < 0.5, 0.2, 0.8)
        return result

    env = SimpleNamespace(
        num_envs=3, device="cpu", dt=0.02,
        robot_config=SimpleNamespace(anchor_body_index=0),
        motion_lib=SimpleNamespace(
            get_motion_positions=positions,
            get_motion_length=lambda ids: torch.full_like(ids, 2.0, dtype=torch.float),
        ),
        motion_manager=SimpleNamespace(
            motion_ids=torch.zeros(3, dtype=torch.long), motion_times=torch.zeros(3)
        ),
    )
    inputs = make_task_inputs(3, reference=fake_reference(env), dt=0.02)
    ordinary = MimicControl(
        MimicControlConfig(reference_height_smoothing_seconds=0.1), inputs
    )
    grace = MimicControl(MimicControlConfig(
        reference_height_smoothing_seconds=0.1, low_pose_grace_seconds=0.5,
    ), inputs)

    def smoothed(control, times, env_ids=None):
        # The smoother reads the clip assignment through the query.
        env.motion_manager.motion_times = times.clone()
        return control._smoothed_reference_height(env_ids)

    times = torch.tensor([0.60, 1.00, 1.10])
    expected = smoothed(ordinary, times - 0.5)
    actual = smoothed(grace, times)
    torch.testing.assert_close(actual, expected)
    assert (actual < 0.5).tolist() == [True, True, False]
    assert (smoothed(ordinary, times) >= 0.5).all()
    smoothed(grace, times + 0.5)
    torch.testing.assert_close(smoothed(grace, times), actual)
    torch.testing.assert_close(smoothed(grace, times, torch.tensor([1, 2])), actual[1:])


@pytest.mark.parametrize("grace", [-0.1, float("nan"), float("inf")])
def test_invalid_grace(grace):
    with pytest.raises(ValueError, match="low_pose_grace_seconds"):
        MimicControl(MimicControlConfig(
            reference_height_smoothing_seconds=0.1, low_pose_grace_seconds=grace,
        ), make_task_inputs(1))


def test_grace_requires_height_sampling():
    with pytest.raises(ValueError, match="requires reference height smoothing"):
        MimicControl(
            MimicControlConfig(low_pose_grace_seconds=0.5), make_task_inputs(1)
        )
