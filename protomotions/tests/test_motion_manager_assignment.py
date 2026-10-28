# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The clip-assignment API outside code uses to pin motions on a MotionManager."""

from __future__ import annotations

import pytest
import torch

from protomotions.envs.motion_manager.config import MotionManagerConfig
from protomotions.envs.motion_manager.motion_manager import MotionManager


class _MotionLib:
    def __init__(self):
        self.motion_weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        self.motion_lengths = torch.tensor([1.0, 2.0, 3.0, 4.0])
        self.motion_file = "motions.yaml"

    def num_motions(self):
        return len(self.motion_weights)


def _manager(num_envs=3, fixed_motion_ids_per_env=None):
    return MotionManager(
        MotionManagerConfig(init_start_prob=0.0),
        num_envs=num_envs,
        env_dt=0.25,
        device=torch.device("cpu"),
        motion_lib=_MotionLib(),
        fixed_motion_ids_per_env=fixed_motion_ids_per_env,
        generator=None,
    )


def test_get_assignment_returns_clones_that_later_writes_cannot_change():
    manager = _manager()
    manager.motion_ids[:] = torch.tensor([1, 2, 3])
    manager.motion_times[:] = torch.tensor([0.5, 0.25, 0.75])

    ids, times = manager.get_assignment()

    assert torch.equal(ids, torch.tensor([1, 2, 3]))
    assert torch.equal(times, torch.tensor([0.5, 0.25, 0.75]))
    assert ids.data_ptr() != manager.motion_ids.data_ptr()
    assert times.data_ptr() != manager.motion_times.data_ptr()
    manager.motion_ids[0] = 0
    manager.motion_times[0] = 0.0
    assert ids[0] == 1 and times[0] == 0.5


def test_set_assignment_writes_in_place_and_keeps_tensor_identity():
    manager = _manager()
    ids, times = manager.motion_ids, manager.motion_times
    # Another object holding the tensors (a control component, the recorder)
    # must see the new values through its reference.
    held = (manager.motion_ids, manager.motion_times)

    manager.set_assignment(torch.tensor([3, 0, 2]), torch.tensor([0.1, 0.2, 0.3]))

    assert manager.motion_ids is ids and manager.motion_times is times
    assert torch.equal(held[0], torch.tensor([3, 0, 2]))
    assert torch.equal(held[1], torch.tensor([0.1, 0.2, 0.3]))


def test_set_assignment_rejects_wrong_shapes_without_a_partial_write():
    manager = _manager()
    manager.set_assignment(torch.tensor([1, 1, 1]), torch.tensor([1.0, 1.0, 1.0]))

    with pytest.raises(ValueError, match=r"motion_ids must have shape \(3,\)"):
        manager.set_assignment(torch.tensor([0, 0]), torch.zeros(3))
    with pytest.raises(ValueError, match=r"motion_times must have shape \(3,\)"):
        manager.set_assignment(torch.tensor([0, 0, 0]), torch.zeros(3, 1))

    assert torch.equal(manager.motion_ids, torch.tensor([1, 1, 1]))
    assert torch.equal(manager.motion_times, torch.tensor([1.0, 1.0, 1.0]))


def test_assign_touches_only_the_given_rows_and_fields():
    manager = _manager()
    ids, times = manager.motion_ids, manager.motion_times
    manager.set_assignment(torch.tensor([0, 1, 2]), torch.tensor([1.0, 2.0, 3.0]))

    manager.assign(torch.tensor([2, 0]), motion_ids=torch.tensor([3, 1]))
    assert torch.equal(manager.motion_ids, torch.tensor([1, 1, 3]))
    assert torch.equal(manager.motion_times, torch.tensor([1.0, 2.0, 3.0]))

    manager.assign(torch.tensor([1]), motion_times=torch.tensor([0.0]))
    assert torch.equal(manager.motion_ids, torch.tensor([1, 1, 3]))
    assert torch.equal(manager.motion_times, torch.tensor([1.0, 0.0, 3.0]))

    manager.assign(
        torch.tensor([0, 2]),
        motion_ids=torch.tensor([2, 0]),
        motion_times=torch.tensor([0.5, 0.25]),
    )
    assert torch.equal(manager.motion_ids, torch.tensor([2, 1, 0]))
    assert torch.equal(manager.motion_times, torch.tensor([0.5, 0.0, 0.25]))
    assert manager.motion_ids is ids and manager.motion_times is times


def test_assign_rejects_missing_or_mismatched_values():
    manager = _manager()
    env_ids = torch.tensor([0, 2])

    with pytest.raises(ValueError, match="motion_ids, motion_times or both"):
        manager.assign(env_ids)
    with pytest.raises(ValueError, match=r"motion_ids must match env_ids.shape \(2,\)"):
        manager.assign(env_ids, motion_ids=torch.tensor([1]))
    with pytest.raises(
        ValueError, match=r"motion_times must match env_ids.shape \(2,\)"
    ):
        manager.assign(
            env_ids, motion_ids=torch.tensor([1, 2]), motion_times=torch.zeros(3)
        )

    assert torch.equal(manager.motion_ids, torch.zeros(3, dtype=torch.long))
    assert torch.equal(manager.motion_times, torch.zeros(3))


def test_fixed_motion_ids_per_env_is_read_only_and_may_be_none():
    assert _manager().fixed_motion_ids_per_env is None

    fixed = torch.tensor([2, -1, 1])
    pinned = _manager(fixed_motion_ids_per_env=fixed)
    assert torch.equal(pinned.fixed_motion_ids_per_env, fixed)
    with pytest.raises(AttributeError):
        pinned.fixed_motion_ids_per_env = torch.tensor([0, 0, 0])
