# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure-tensor termination kernels in
protomotions.envs.terminations (base predicates + task scalar kernels).
"""

from __future__ import annotations


import torch

from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.terminations.base import (
    check_max_length_term,
    named_contact_fall_termination,
)
from protomotions.envs.terminations.rule import TerminationRule
from protomotions.envs.terminations.task import (
    path_distance,
    path_height_error,
    steering_direction_alignment,
    steering_speed_error,
)


# ---------- named_contact_fall_termination ------------------------------------
# Bodies: 0, 1 = feet (allowed support), 2 = torso, 3 = head. The contact matrix
# carries only the forbidden rows (torso, head) against two partners, with
# terrain deliberately in column 1 so nothing relies on a fixed terrain index.

BODY_INDEX = {"torso": 0, "head": 1}
PARTNER_INDEX = {"robot/foot_l": 0, "terrain": 1}
FORBIDDEN = ["torso", "head"]
FORBIDDEN_IDS = torch.tensor([2, 3], dtype=torch.long)


def _positions(torso_z: float, head_z: float, envs: int = 1) -> torch.Tensor:
    pos = torch.zeros(envs, 4, 3)
    pos[:, 0, 2], pos[:, 1, 2] = 0.05, 0.05  # feet always low: never a fall
    pos[:, 2, 2], pos[:, 3, 2] = torso_z, head_z
    return pos


def _terrain_forces(torso: list[float], head: list[float]) -> torch.Tensor:
    forces = torch.zeros(len(torso), 2, 2, 3)
    forces[:, 0, 1, 2] = torch.tensor(torso)
    forces[:, 1, 1, 2] = torch.tensor(head)
    return forces


def test_named_fall_only_flags_envs_whose_forbidden_rows_touch_terrain():
    # Env 0: nothing on terrain. Env 1: torso. Env 2: head.
    forces = _terrain_forces(torso=[0.0, 5.0, 0.0], head=[0.0, 0.0, 5.0])
    pos = _positions(torso_z=0.05, head_z=0.05, envs=3)

    result = named_contact_fall_termination(
        rigid_body_pos=pos,
        contact_forces=forces,
        contact_body_index=BODY_INDEX,
        contact_partner_index=PARTNER_INDEX,
        ground_heights=torch.zeros(3),
        progress_buf=torch.tensor([5, 5, 5]),
        body_ids=FORBIDDEN_IDS,
        body_names=FORBIDDEN,
        partner_names=["terrain"],
        termination_height=0.1,
    )

    assert torch.equal(result, torch.tensor([False, True, True]))


def test_named_fall_skips_first_two_progress_steps():
    forces = _terrain_forces(torso=[5.0], head=[0.0])
    pos = _positions(torso_z=0.05, head_z=1.0)
    for step, expected in ((0, False), (1, False), (2, True)):
        result = named_contact_fall_termination(
            rigid_body_pos=pos,
            contact_forces=forces,
            contact_body_index=BODY_INDEX,
            contact_partner_index=PARTNER_INDEX,
            ground_heights=torch.zeros(1),
            progress_buf=torch.tensor([step]),
            body_ids=FORBIDDEN_IDS,
            body_names=FORBIDDEN,
            partner_names=["terrain"],
            termination_height=0.1,
        )
        assert torch.equal(result, torch.tensor([expected])), step


def test_named_fall_counts_only_the_named_partner_columns():
    # Torso presses hard on another robot link (column 0) but not on terrain.
    forces = torch.zeros(1, 2, 2, 3)
    forces[0, 0, 0, 0] = 50.0
    pos = _positions(torso_z=0.05, head_z=1.0)
    kwargs = dict(
        ground_heights=torch.zeros(1),
        progress_buf=torch.tensor([5]),
        body_ids=FORBIDDEN_IDS,
        body_names=FORBIDDEN,
        termination_height=0.1,
    )
    assert torch.equal(
        named_contact_fall_termination(
            rigid_body_pos=pos,
            contact_forces=forces,
            contact_body_index=BODY_INDEX,
            contact_partner_index=PARTNER_INDEX,
            partner_names=["terrain"],
            **kwargs,
        ),
        torch.tensor([False]),
    )
    assert torch.equal(
        named_contact_fall_termination(
            rigid_body_pos=pos,
            contact_forces=forces,
            contact_body_index=BODY_INDEX,
            contact_partner_index=PARTNER_INDEX,
            partner_names=["robot/foot_l"],
            **kwargs,
        ),
        torch.tensor([True]),
    )


def test_named_fall_does_not_mutate_inputs():
    forces = _terrain_forces(torso=[5.0], head=[0.0])
    pos = _positions(torso_z=0.05, head_z=1.0)
    forces_clone, pos_clone = forces.clone(), pos.clone()

    named_contact_fall_termination(
        rigid_body_pos=pos,
        contact_forces=forces,
        contact_body_index=BODY_INDEX,
        contact_partner_index=PARTNER_INDEX,
        ground_heights=torch.zeros(1),
        progress_buf=torch.tensor([5]),
        body_ids=FORBIDDEN_IDS,
        body_names=FORBIDDEN,
        partner_names=["terrain"],
        termination_height=0.1,
    )

    assert torch.equal(forces, forces_clone) and torch.equal(pos, pos_clone)


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


# ---------- named_contact_fall_termination: contact AND height ----------------


def test_named_fall_requires_both_contact_and_low_height():
    forces_touching = _terrain_forces(torso=[5.0], head=[0.0])
    forces_free = torch.zeros(1, 2, 2, 3)
    kwargs = dict(
        ground_heights=torch.zeros(1),
        progress_buf=torch.tensor([5]),
        body_ids=FORBIDDEN_IDS,
        body_names=FORBIDDEN,
        partner_names=["terrain"],
        termination_height=0.1,
    )
    # Torso touches terrain but every forbidden body is high: no fall.
    assert torch.equal(
        named_contact_fall_termination(
            rigid_body_pos=_positions(torso_z=1.0, head_z=1.0),
            contact_forces=forces_touching,
            contact_body_index=BODY_INDEX,
            contact_partner_index=PARTNER_INDEX,
            **kwargs,
        ),
        torch.tensor([False]),
    )
    # Torso is low but nothing touches terrain: no fall.
    assert torch.equal(
        named_contact_fall_termination(
            rigid_body_pos=_positions(torso_z=0.05, head_z=1.0),
            contact_forces=forces_free,
            contact_body_index=BODY_INDEX,
            contact_partner_index=PARTNER_INDEX,
            **kwargs,
        ),
        torch.tensor([False]),
    )
    # Both: fall.
    assert torch.equal(
        named_contact_fall_termination(
            rigid_body_pos=_positions(torso_z=0.05, head_z=1.0),
            contact_forces=forces_touching,
            contact_body_index=BODY_INDEX,
            contact_partner_index=PARTNER_INDEX,
            **kwargs,
        ),
        torch.tensor([True]),
    )


def test_named_fall_keeps_contact_and_height_per_env():
    # Env 0: contact + low torso. Env 1: contact, nothing low. Env 2: low, no contact.
    forces = _terrain_forces(torso=[5.0, 5.0, 0.0], head=[0.0, 0.0, 0.0])
    pos = _positions(torso_z=1.0, head_z=1.0, envs=3)
    pos[0, 2, 2], pos[2, 2, 2] = 0.05, 0.05

    result = named_contact_fall_termination(
        rigid_body_pos=pos,
        contact_forces=forces,
        contact_body_index=BODY_INDEX,
        contact_partner_index=PARTNER_INDEX,
        ground_heights=torch.zeros(3),
        progress_buf=torch.tensor([5, 5, 5]),
        body_ids=FORBIDDEN_IDS,
        body_names=FORBIDDEN,
        partner_names=["terrain"],
        termination_height=0.1,
    )

    assert torch.equal(result, torch.tensor([True, False, False]))


def test_named_fall_uses_each_env_ground_height_independently():
    """Heights are measured above each environment's own ground level."""
    forces = _terrain_forces(torso=[5.0, 5.0], head=[0.0, 0.0])
    pos = _positions(torso_z=0.75, head_z=2.0, envs=2)  # same absolute heights

    result = named_contact_fall_termination(
        rigid_body_pos=pos,
        contact_forces=forces,
        contact_body_index=BODY_INDEX,
        contact_partner_index=PARTNER_INDEX,
        ground_heights=torch.tensor([0.0, 0.5]),
        progress_buf=torch.tensor([5, 5]),
        body_ids=FORBIDDEN_IDS,
        body_names=FORBIDDEN,
        partner_names=["terrain"],
        termination_height=0.5,
    )

    # Env 0: torso 0.75 above ground, safe. Env 1: 0.25 above ground, fall.
    assert torch.equal(result, torch.tensor([False, True]))


def test_named_fall_resolves_rows_and_kinematic_ids_separately():
    """Rows address the compact matrix; body_ids address full kinematics."""
    # Matrix holds only the head (row 0); kinematics put the head at index 3.
    forces = torch.zeros(1, 1, 1, 3)
    forces[0, 0, 0, 2] = 5.0
    pos = _positions(torso_z=1.0, head_z=0.05)

    result = named_contact_fall_termination(
        rigid_body_pos=pos,
        contact_forces=forces,
        contact_body_index={"head": 0},
        contact_partner_index={"terrain": 0},
        ground_heights=torch.zeros(1),
        progress_buf=torch.tensor([5]),
        body_ids=torch.tensor([3]),
        body_names=["head"],
        partner_names=["terrain"],
        termination_height=0.1,
    )

    assert torch.equal(result, torch.tensor([True]))


def test_named_fall_binds_its_state_inputs_by_default():
    """The recipe supplies only the selection; the state inputs are defaults."""
    component = MdpComponent.from_kernel(
        named_contact_fall_termination,
        required_keys={
            "contact_body_index": FORBIDDEN,
            "contact_partner_index": ["terrain"],
        },
        body_ids=FORBIDDEN_IDS,
        body_names=FORBIDDEN,
        partner_names=["terrain"],
    )
    assert component.get_bindings_dict() == {
        "rigid_body_pos": "current.rigid_body_pos",
        "contact_forces": "contact_forces",
        "contact_body_index": "contact_body_index",
        "contact_partner_index": "contact_partner_index",
        "ground_heights": "ground_heights",
        "progress_buf": "progress_buf",
    }
    assert component.get_params()["termination_height"] == 0.15
    assert TerminationRule(component).threshold is None


# ---------- task.py: path_distance / path_height_error -----------------------
# The task kernels are scalar measurements; ``TerminationRule(component,
# threshold, fail_above=...)`` cuts them, so each old boolean check is a rule
# over the kernel's value here.


def _rule(kernel, threshold, fail_above=True):
    return TerminationRule(MdpComponent.from_kernel(kernel), threshold, fail_above)


def test_path_distance_is_zero_until_min_progress_is_exceeded():
    head = torch.zeros(2, 3)
    target = torch.tensor([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])

    # Early progress: the kernel reads zero, so no rule can fire on it.
    early = path_distance(head, target, torch.tensor([5, 5]), min_progress=10)
    assert torch.equal(early, torch.zeros(2))
    assert not _rule(path_distance, 2.0).compare(early).any()
    # Late progress: the 5 m distance is read and the 2 m rule fires.
    late = path_distance(head, target, torch.tensor([20, 20]), min_progress=10)
    assert torch.equal(late, torch.tensor([5.0, 5.0]))
    assert _rule(path_distance, 2.0).compare(late).all()


def test_path_distance_min_progress_boundary_is_strict():
    head = torch.zeros(2, 3)
    target = torch.tensor([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])

    distance = path_distance(head, target, torch.tensor([10, 11]), min_progress=10)

    assert torch.equal(distance, torch.tensor([0.0, 5.0]))


def test_path_distance_rule_is_strict_at_the_threshold():
    head = torch.zeros(1, 3)
    # 3-4-5 right triangle: head→target distance == 5.0.
    target = torch.tensor([[3.0, 4.0, 0.0]])
    distance = path_distance(head, target, torch.tensor([100]))

    assert torch.equal(distance, torch.tensor([5.0]))
    assert _rule(path_distance, 4.99).compare(distance).tolist() == [True]
    assert _rule(path_distance, 5.0).compare(distance).tolist() == [False]
    assert _rule(path_distance, 5.01).compare(distance).tolist() == [False]


def test_path_height_error_only_z_axis_matters():
    """A large XY offset does not register as a height error."""
    head = torch.tensor([[100.0, 100.0, 1.0]])
    progress = torch.tensor([100])

    assert torch.equal(
        path_height_error(head, torch.tensor([[0.0, 0.0, 1.0]]), progress),
        torch.tensor([0.0]),
    )
    assert torch.equal(
        path_height_error(head, torch.tensor([[0.0, 0.0, 2.0]]), progress),
        torch.tensor([1.0]),
    )


def test_path_height_error_min_progress_and_rule_boundaries_are_strict():
    head = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    target = torch.tensor([[0.0, 0.0, 1.6], [0.0, 0.0, 1.6], [0.0, 0.0, 1.5]])
    progress = torch.tensor([10, 11, 11])

    error = path_height_error(head, target, progress, min_progress=10)

    assert torch.allclose(error, torch.tensor([0.0, 0.6, 0.5]))
    assert _rule(path_height_error, 0.5).compare(error).tolist() == [
        False,
        True,
        False,
    ]


# ---------- task.py: steering_speed_error / steering_direction_alignment -----


def _steering(root, prev, tar_dir, tar_speed, dt):
    return (
        steering_speed_error(root, prev, tar_speed, dt),
        steering_direction_alignment(root, prev, tar_dir, dt),
    )


def test_steering_kernels_pass_when_speed_and_direction_match():
    """Constant velocity along +x at 1 m/s, target dir (1, 0) and speed 1."""
    speed_error, alignment = _steering(
        torch.tensor([[0.1, 0.0, 0.0]]),
        torch.zeros(1, 3),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([1.0]),
        dt=0.1,
    )

    assert torch.allclose(speed_error, torch.tensor([0.0]), atol=1e-6)
    assert torch.allclose(alignment, torch.tensor([1.0]), atol=1e-5)
    assert not _rule(steering_speed_error, 0.1).compare(speed_error).any()
    assert not _rule(steering_direction_alignment, 0.95, False).compare(alignment).any()


def test_steering_speed_error_measures_the_planar_speed_mismatch():
    # dt = 0.1, delta = 0.5 m → speed 5 m/s against a 1 m/s target.
    speed_error, _ = _steering(
        torch.tensor([[0.5, 0.0, 0.0]]),
        torch.zeros(1, 3),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([1.0]),
        dt=0.1,
    )

    assert torch.allclose(speed_error, torch.tensor([4.0]))
    assert _rule(steering_speed_error, 0.5).compare(speed_error).tolist() == [True]


def test_steering_direction_alignment_is_the_heading_dot_product():
    # Move along +y while the target direction is +x.
    _, alignment = _steering(
        torch.tensor([[0.0, 0.1, 0.0]]),
        torch.zeros(1, 3),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([1.0]),
        dt=0.1,
    )

    assert torch.allclose(alignment, torch.tensor([0.0]), atol=1e-6)
    off_course = _rule(steering_direction_alignment, 0.5, False).compare(alignment)
    assert off_course.tolist() == [True]


def test_steering_direction_alignment_passes_a_stationary_robot():
    # speed = 0.01 m / 0.1 s = 0.1: at the guard, not strictly > 0.1, so the
    # heading is not judged and the alignment reads a passing 1.0.
    _, alignment = _steering(
        torch.tensor([[0.0, 0.01, 0.0]]),
        torch.zeros(1, 3),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([1.0]),
        dt=0.1,
    )

    assert torch.equal(alignment, torch.tensor([1.0]))
    assert not _rule(steering_direction_alignment, 0.99, False).compare(alignment).any()


def test_steering_rules_or_together_per_env_like_the_old_boolean_check():
    root = torch.tensor(
        [
            [0.1, 0.0, 0.0],  # matches speed and direction
            [0.3, 0.0, 0.0],  # too fast
            [0.0, 0.1, 0.0],  # wrong direction
        ]
    )
    speed_error, alignment = _steering(
        root,
        torch.zeros_like(root),
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        torch.ones(3),
        dt=0.1,
    )
    too_fast = _rule(steering_speed_error, 0.2).compare(speed_error)
    off_course = _rule(steering_direction_alignment, 0.5, False).compare(alignment)

    assert torch.equal(too_fast, torch.tensor([False, True, False]))
    assert torch.equal(off_course, torch.tensor([False, False, True]))
    assert torch.equal(too_fast | off_course, torch.tensor([False, True, True]))
