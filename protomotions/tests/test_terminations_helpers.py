# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure-tensor termination predicates in
protomotions.envs.terminations (base + task helpers).
"""

from __future__ import annotations

import math

import pytest
import torch

from protomotions.envs.terminations.base import (
    check_fall_contact_term,
    check_height_term,
    check_max_length_term,
    combine_fall_termination,
    contact_termination,
    fall_termination,
    height_termination,
    threshold_termination,
)
from protomotions.envs.terminations.task import (
    check_path_distance_term,
    check_path_height_term,
    check_steering_velocity_error,
)


# ---------- check_fall_contact_term -------------------------------------------


def test_check_fall_contact_term_only_flags_envs_with_disallowed_contacts():
    # 3 envs × 4 bodies. Bodies 0, 1 are feet (allowed contact); body 2 = torso, body 3 = head.
    contacts = torch.tensor(
        [
            [True, True, False, False],   # only feet — should not terminate
            [True, False, True, False],   # torso contact — should terminate
            [False, False, False, True],  # head contact — should terminate
        ]
    )
    allowed = torch.tensor([0, 1], dtype=torch.long)
    progress = torch.tensor([5, 5, 5])

    result = check_fall_contact_term(contacts, allowed, progress)

    assert torch.equal(result, torch.tensor([False, True, True]))


def test_check_fall_contact_term_skips_first_two_progress_steps():
    """progress_buf <= 1 should never terminate even with bad contact."""
    contacts = torch.tensor([[False, False, True]])
    allowed = torch.tensor([0], dtype=torch.long)

    assert torch.equal(
        check_fall_contact_term(contacts, allowed, torch.tensor([0])),
        torch.tensor([False]),
    )
    assert torch.equal(
        check_fall_contact_term(contacts, allowed, torch.tensor([1])),
        torch.tensor([False]),
    )
    assert torch.equal(
        check_fall_contact_term(contacts, allowed, torch.tensor([2])),
        torch.tensor([True]),
    )


def test_check_fall_contact_term_does_not_mutate_input_contacts():
    contacts = torch.tensor([[True, True, False]])
    contacts_clone = contacts.clone()
    allowed = torch.tensor([0], dtype=torch.long)

    _ = check_fall_contact_term(contacts, allowed, torch.tensor([5]))

    # The internal mask must be on a clone — input should be untouched.
    assert torch.equal(contacts, contacts_clone)


def test_check_fall_contact_term_handles_no_allowed_contact_bodies():
    contacts = torch.tensor(
        [
            [False, False, False],
            [False, True, False],
        ]
    )
    allowed = torch.tensor([], dtype=torch.long)
    progress = torch.tensor([5, 5])

    result = check_fall_contact_term(contacts, allowed, progress)

    assert torch.equal(result, torch.tensor([False, True]))


# ---------- check_height_term --------------------------------------------------


def test_check_height_term_flags_bodies_below_threshold():
    # 2 envs, 3 bodies (z values).
    rigid_body_pos = torch.tensor(
        [
            [[0.0, 0.0, 0.5], [0.0, 0.0, 1.0], [0.0, 0.0, 1.5]],   # all above
            [[0.0, 0.0, 0.05], [0.0, 0.0, 1.0], [0.0, 0.0, 1.5]],  # body 0 below
        ]
    )
    termination_heights = torch.tensor([0.1, 0.5, 0.5])
    allowed = torch.tensor([], dtype=torch.long)

    result = check_height_term(rigid_body_pos, termination_heights, allowed)

    assert torch.equal(result, torch.tensor([False, True]))


def test_check_height_term_excludes_allowed_bodies_from_violation():
    """A body listed in non_termination_body_ids never triggers termination
    even if below threshold."""
    rigid_body_pos = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    termination_heights = torch.tensor([0.1, 0.5])
    # Body 0 is below threshold but is allowed — should not terminate.
    allowed = torch.tensor([0], dtype=torch.long)

    result = check_height_term(rigid_body_pos, termination_heights, allowed)

    assert torch.equal(result, torch.tensor([False]))


def test_check_height_term_uses_strict_less_than_threshold():
    rigid_body_pos = torch.tensor(
        [
            [[0.0, 0.0, 0.10], [0.0, 0.0, 0.50]],
            [[0.0, 0.0, 0.09], [0.0, 0.0, 0.50]],
        ]
    )
    termination_heights = torch.tensor([0.10, 0.50])
    allowed = torch.tensor([], dtype=torch.long)

    result = check_height_term(rigid_body_pos, termination_heights, allowed)

    assert torch.equal(result, torch.tensor([False, True]))


# ---------- check_max_length_term ---------------------------------------------


def test_check_max_length_term_triggers_at_one_step_before_max():
    progress = torch.tensor([0, 5, 9, 10, 11])
    result = check_max_length_term(progress, max_episode_length=10.0)
    # progress >= max - 1 = 9 → step 9, 10, 11 all terminate.
    assert torch.equal(result, torch.tensor([False, False, True, True, True]))


def test_check_max_length_term_accepts_fractional_episode_lengths():
    progress = torch.tensor([7.0, 7.4, 7.5, 8.0])

    result = check_max_length_term(progress, max_episode_length=8.5)

    assert torch.equal(result, torch.tensor([False, False, True, True]))


# ---------- combine_fall_termination + fall_termination wrapper ---------------


def test_combine_fall_termination_requires_both_contact_and_height():
    """The combined check uses AND: contact alone or height alone shouldn't terminate."""
    # Single env, 3 bodies.
    progress = torch.tensor([5])
    allowed = torch.tensor([0], dtype=torch.long)
    termination_heights = torch.tensor([0.1, 0.1, 0.1])

    # Contact only on body 1, but body 1 is above threshold.
    contacts_only = torch.tensor([[False, True, False]])
    pos_above = torch.tensor([[[0, 0, 1.0], [0, 0, 1.0], [0, 0, 1.0]]], dtype=torch.float)
    assert torch.equal(
        combine_fall_termination(contacts_only, pos_above, termination_heights, allowed, progress),
        torch.tensor([False]),
    )

    # Height violation on body 1, but no contact on body 1.
    no_contacts = torch.tensor([[False, False, False]])
    pos_below = torch.tensor([[[0, 0, 1.0], [0, 0, 0.0], [0, 0, 1.0]]], dtype=torch.float)
    assert torch.equal(
        combine_fall_termination(no_contacts, pos_below, termination_heights, allowed, progress),
        torch.tensor([False]),
    )

    # Both contact and height violation on body 1 — must terminate.
    assert torch.equal(
        combine_fall_termination(contacts_only, pos_below, termination_heights, allowed, progress),
        torch.tensor([True]),
    )


def test_combine_fall_termination_keeps_contact_and_height_per_env():
    contacts = torch.tensor(
        [
            [False, True, False],   # contact and low torso: terminate
            [False, True, False],   # contact but no low body: do not terminate
            [False, False, False],  # low torso but no contact: do not terminate
        ]
    )
    pos = torch.tensor(
        [
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.05], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.50], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.05], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float,
    )
    termination_heights = torch.tensor([0.1, 0.1, 0.1])
    allowed = torch.tensor([0], dtype=torch.long)
    progress = torch.tensor([5, 5, 5])

    result = combine_fall_termination(
        contacts, pos, termination_heights, allowed, progress
    )

    assert torch.equal(result, torch.tensor([True, False, False]))


def test_fall_termination_wrapper_adjusts_threshold_by_ground_height():
    """fall_termination(termination_height=0.5) on flat ground (h=0) is the
    same as combine_fall_termination with all body thresholds = 0.5; on a 1.0
    elevated platform the bodies must clear 1.5 to avoid termination."""
    contacts = torch.tensor([[False, True, False]])
    progress = torch.tensor([5])
    allowed = torch.tensor([0], dtype=torch.long)

    # On flat ground, body 1 at z=0.4 with termination_height=0.5 → below.
    pos = torch.tensor([[[0, 0, 1.0], [0, 0, 0.4], [0, 0, 1.0]]], dtype=torch.float)
    assert torch.equal(
        fall_termination(pos, contacts, torch.tensor([0.0]), 0.5, allowed, progress),
        torch.tensor([True]),
    )

    # On a 1.0-high platform, the SAME body z=0.4 is now above ground locally
    # (relative threshold = 1.5) but absolute z=0.4 is still below. The wrapper
    # adds ground_height to threshold, so threshold = 1.5; z=0.4 < 1.5 → still
    # below threshold → still terminates.
    assert torch.equal(
        fall_termination(pos, contacts, torch.tensor([1.0]), 0.5, allowed, progress),
        torch.tensor([True]),
    )

    # Lift ALL bodies above the elevated threshold (z=2.0 > 1.5) → no
    # termination. check_height_term inspects every non-allowed body, so even
    # body 2 (no contact) must clear the threshold for the height-side AND
    # to fail.
    pos_lifted = torch.tensor(
        [[[0, 0, 2.0], [0, 0, 2.0], [0, 0, 2.0]]], dtype=torch.float
    )
    assert torch.equal(
        fall_termination(
            pos_lifted, contacts, torch.tensor([1.0]), 0.5, allowed, progress
        ),
        torch.tensor([False]),
    )


def test_fall_termination_uses_each_env_ground_height_independently():
    contacts = torch.tensor(
        [
            [False, True],
            [False, True],
        ]
    )
    # Body 1 has the same absolute height in both envs.
    pos = torch.tensor(
        [
            [[0.0, 0.0, 2.0], [0.0, 0.0, 0.75]],
            [[0.0, 0.0, 2.0], [0.0, 0.0, 0.75]],
        ],
        dtype=torch.float,
    )
    allowed = torch.tensor([0], dtype=torch.long)
    progress = torch.tensor([5, 5])

    result = fall_termination(
        pos,
        contacts,
        ground_heights=torch.tensor([0.0, 0.5]),
        termination_height=0.5,
        non_termination_contact_body_ids=allowed,
        progress_buf=progress,
    )

    # Env 0 threshold is 0.5, so z=0.75 is safe. Env 1 threshold is 1.0.
    assert torch.equal(result, torch.tensor([False, True]))


def test_height_termination_wrapper_creates_per_body_threshold():
    pos = torch.tensor(
        [
            [[0, 0, 0.05], [0, 0, 1.0]],
            [[0, 0, 0.30], [0, 0, 1.0]],
        ],
        dtype=torch.float,
    )
    allowed = torch.tensor([], dtype=torch.long)
    ground = torch.tensor([0.0, 0.0])
    # Threshold = 0.1 for all bodies. Env 0 body 0 at 0.05 < 0.1 → terminate.
    result = height_termination(pos, ground, 0.1, allowed)
    assert torch.equal(result, torch.tensor([True, False]))


def test_height_termination_uses_each_env_ground_height_independently():
    pos = torch.tensor(
        [
            [[0.0, 0.0, 0.75], [0.0, 0.0, 2.0]],
            [[0.0, 0.0, 0.75], [0.0, 0.0, 2.0]],
        ],
        dtype=torch.float,
    )
    allowed = torch.tensor([], dtype=torch.long)
    ground = torch.tensor([0.0, 0.5])

    result = height_termination(pos, ground, 0.5, allowed)

    assert torch.equal(result, torch.tensor([False, True]))


def test_contact_termination_wrapper_matches_check_fall_contact_term():
    contacts = torch.tensor([[True, False, False], [False, True, False]])
    progress = torch.tensor([5, 5])
    allowed = torch.tensor([0], dtype=torch.long)

    expected = check_fall_contact_term(contacts, allowed, progress)
    result = contact_termination(contacts, allowed, progress)
    assert torch.equal(result, expected)


# ---------- threshold_termination ---------------------------------------------


def test_threshold_termination_greater_than_default_is_strict():
    values = torch.tensor([0.5, 1.0, 1.5])
    result = threshold_termination(values, threshold=1.0)
    # Strict greater-than: 1.0 itself does not trigger.
    assert torch.equal(result, torch.tensor([False, False, True]))


def test_threshold_termination_less_than_mode_flips_comparison():
    values = torch.tensor([0.5, 1.0, 1.5])
    result = threshold_termination(values, threshold=1.0, greater_than=False)
    assert torch.equal(result, torch.tensor([True, False, False]))


def test_threshold_termination_reduces_high_dim_input_to_per_env_mean():
    values = torch.tensor(
        [
            [1.0, 1.0, 1.0],   # mean 1.0 — not greater than 0.5
            [0.0, 0.0, 0.0],   # mean 0.0 — not greater than 0.5
            [2.0, 0.0, 1.0],   # mean 1.0 — greater than 0.5
        ]
    )
    result = threshold_termination(values, threshold=0.5)
    assert torch.equal(result, torch.tensor([True, False, True]))


def test_threshold_termination_reduces_all_non_batch_dimensions():
    values = torch.tensor(
        [
            [[[2.0], [2.0]], [[0.0], [0.0]]],  # mean 1.0
            [[[0.0], [0.0]], [[0.0], [0.0]]],  # mean 0.0
        ]
    )

    result = threshold_termination(values, threshold=0.75)

    assert torch.equal(result, torch.tensor([True, False]))


# ---------- task.py: check_path_distance_term --------------------------------


def test_check_path_distance_term_only_after_min_progress():
    head = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    target = torch.tensor([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    fail_dist = 2.0  # head is 5m from target → would fail dist check

    early_progress = torch.tensor([5, 5])
    late_progress = torch.tensor([20, 20])

    # Early progress: NEVER terminate even though distance check would fail.
    assert torch.equal(
        check_path_distance_term(head, target, fail_dist, early_progress, min_progress=10),
        torch.tensor([False, False]),
    )
    # Late progress: distance check kicks in.
    assert torch.equal(
        check_path_distance_term(head, target, fail_dist, late_progress, min_progress=10),
        torch.tensor([True, True]),
    )


def test_check_path_distance_term_min_progress_boundary_is_strict():
    head = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    target = torch.tensor([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    progress = torch.tensor([10, 11])

    result = check_path_distance_term(
        head, target, fail_dist=2.0, progress_buf=progress, min_progress=10
    )

    assert torch.equal(result, torch.tensor([False, True]))


def test_check_path_distance_term_uses_squared_distance():
    head = torch.tensor([[0.0, 0.0, 0.0]])
    # 3-4-5 right triangle: head→target distance == 5.0.
    target = torch.tensor([[3.0, 4.0, 0.0]])
    progress = torch.tensor([100])

    # fail_dist=4.99 → squared 24.9, dist² = 25 > 24.9 → terminate.
    assert torch.equal(
        check_path_distance_term(head, target, 4.99, progress),
        torch.tensor([True]),
    )
    # fail_dist=5.01 → squared 25.1, dist² = 25 < 25.1 → no termination.
    assert torch.equal(
        check_path_distance_term(head, target, 5.01, progress),
        torch.tensor([False]),
    )


def test_check_path_distance_term_is_strict_at_distance_threshold():
    head = torch.tensor([[0.0, 0.0, 0.0]])
    target = torch.tensor([[3.0, 4.0, 0.0]])
    progress = torch.tensor([100])

    result = check_path_distance_term(head, target, 5.0, progress)

    assert torch.equal(result, torch.tensor([False]))


# ---------- task.py: check_path_height_term ----------------------------------


def test_check_path_height_term_only_z_axis_matters():
    """A large XY offset should not trigger height termination."""
    head = torch.tensor([[100.0, 100.0, 1.0]])
    target = torch.tensor([[0.0, 0.0, 1.0]])  # same z
    progress = torch.tensor([100])

    assert torch.equal(
        check_path_height_term(head, target, fail_height_dist=0.5, progress_buf=progress),
        torch.tensor([False]),
    )

    # Now z differs by 1.0.
    target_high = torch.tensor([[0.0, 0.0, 2.0]])
    assert torch.equal(
        check_path_height_term(head, target_high, fail_height_dist=0.5, progress_buf=progress),
        torch.tensor([True]),
    )


def test_check_path_height_term_min_progress_and_threshold_boundaries_are_strict():
    head = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target = torch.tensor(
        [
            [0.0, 0.0, 1.6],
            [0.0, 0.0, 1.6],
            [0.0, 0.0, 1.5],
        ]
    )
    progress = torch.tensor([10, 11, 11])

    result = check_path_height_term(
        head, target, fail_height_dist=0.5, progress_buf=progress, min_progress=10
    )

    assert torch.equal(result, torch.tensor([False, True, False]))


# ---------- task.py: check_steering_velocity_error ---------------------------


def test_check_steering_velocity_error_passes_when_speed_and_direction_match():
    """Constant-velocity along +x at 1 m/s with target dir=(1,0) and target speed=1 → no fail."""
    root = torch.tensor([[0.1, 0.0, 0.0]])
    prev = torch.tensor([[0.0, 0.0, 0.0]])
    tar_dir = torch.tensor([[1.0, 0.0]])
    tar_speed = torch.tensor([1.0])

    fail = check_steering_velocity_error(
        root_pos=root,
        prev_root_pos=prev,
        tar_dir=tar_dir,
        tar_speed=tar_speed,
        dt=0.1,
        speed_tolerance=0.1,
        direction_tolerance=0.95,
    )
    assert torch.equal(fail, torch.tensor([False]))


def test_check_steering_velocity_error_fails_on_speed_mismatch():
    # dt=0.1, delta=0.5 m → speed = 5 m/s, target = 1 m/s, tolerance = 0.5.
    root = torch.tensor([[0.5, 0.0, 0.0]])
    prev = torch.tensor([[0.0, 0.0, 0.0]])
    tar_dir = torch.tensor([[1.0, 0.0]])
    tar_speed = torch.tensor([1.0])

    fail = check_steering_velocity_error(
        root_pos=root,
        prev_root_pos=prev,
        tar_dir=tar_dir,
        tar_speed=tar_speed,
        dt=0.1,
        speed_tolerance=0.5,
        direction_tolerance=0.5,
    )
    assert torch.equal(fail, torch.tensor([True]))


def test_check_steering_velocity_error_fails_on_direction_mismatch():
    # Move along +y but target dir = +x.
    root = torch.tensor([[0.0, 0.1, 0.0]])
    prev = torch.tensor([[0.0, 0.0, 0.0]])
    tar_dir = torch.tensor([[1.0, 0.0]])
    tar_speed = torch.tensor([1.0])

    fail = check_steering_velocity_error(
        root_pos=root,
        prev_root_pos=prev,
        tar_dir=tar_dir,
        tar_speed=tar_speed,
        dt=0.1,
        speed_tolerance=10.0,  # very loose so speed isn't the trigger
        direction_tolerance=0.5,
    )
    assert torch.equal(fail, torch.tensor([True]))


def test_check_steering_velocity_error_skips_direction_check_at_low_speed():
    """When measured speed < 0.1, direction-mismatch should not trigger fail."""
    # Tiny motion so measured speed << 0.1.
    root = torch.tensor([[0.0, 0.001, 0.0]])
    prev = torch.tensor([[0.0, 0.0, 0.0]])
    tar_dir = torch.tensor([[1.0, 0.0]])
    tar_speed = torch.tensor([0.01])

    fail = check_steering_velocity_error(
        root_pos=root,
        prev_root_pos=prev,
        tar_dir=tar_dir,
        tar_speed=tar_speed,
        dt=0.1,
        speed_tolerance=10.0,  # speed check disabled
        direction_tolerance=0.99,  # would fail if speed > 0.1
    )
    # speed = 0.01 m / 0.1s = 0.1 — exactly at the threshold; not strictly > 0.1.
    assert torch.equal(fail, torch.tensor([False]))


def test_check_steering_velocity_error_evaluates_speed_and_direction_per_env():
    root = torch.tensor(
        [
            [0.1, 0.0, 0.0],  # matches speed and direction
            [0.3, 0.0, 0.0],  # too fast
            [0.0, 0.1, 0.0],  # wrong direction
        ]
    )
    prev = torch.zeros_like(root)
    tar_dir = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    tar_speed = torch.tensor([1.0, 1.0, 1.0])

    fail = check_steering_velocity_error(
        root_pos=root,
        prev_root_pos=prev,
        tar_dir=tar_dir,
        tar_speed=tar_speed,
        dt=0.1,
        speed_tolerance=0.2,
        direction_tolerance=0.5,
    )

    assert torch.equal(fail, torch.tensor([False, True, True]))
