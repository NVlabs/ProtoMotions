# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Newton label, shape-axis and per-contact reduction bookkeeping (no Newton import)."""

import pytest
import torch

from protomotions.simulator.newton.contact_utils import (
    contact_force_matrix,
    resolve_world_body_indices,
    resolve_world_projectile_indices,
    shape_axes,
)

# Newton 1.x keeps MJCF paths in body_label and repeats them per replicated world.
G1_LABELS = [
    "g1_29dof/worldbody/pelvis",
    "g1_29dof/worldbody/pelvis/left_hip_pitch_link/left_hip_roll_link/"
    "left_hip_yaw_link/left_knee_link/left_ankle_pitch_link/left_ankle_roll_link",
    "body_2",  # projectile added with add_body(): default label
]
LABELS = G1_LABELS * 2
WORLDS = [0, 0, 0, 1, 1, 1]


def test_robot_bodies_resolve_per_world_by_short_label_in_requested_order():
    # Requested order differs from creation order to catch index/order mix-ups.
    rows = resolve_world_body_indices(
        LABELS, WORLDS, 2, ["left_ankle_roll_link", "pelvis"]
    )
    assert rows == [[1, 0], [4, 3]]


def test_missing_or_duplicated_robot_bodies_fail_loudly():
    with pytest.raises(ValueError, match="lacks robot bodies"):
        resolve_world_body_indices(LABELS, WORLDS, 2, ["pelvis", "head"])
    with pytest.raises(ValueError, match="appears twice"):
        resolve_world_body_indices(["a/x", "b/x"], [0, 0], 1, ["x"])


def test_projectiles_are_the_remaining_world_bodies_in_order():
    robot = [[1, 0], [4, 3]]
    assert resolve_world_projectile_indices(WORLDS, 2, robot, 1) == [[2], [5]]
    with pytest.raises(ValueError, match="expected 0 projectiles"):
        resolve_world_projectile_indices(WORLDS, 2, robot, 0)


def test_shape_axes_select_rows_partners_ground_and_skip_visuals():
    # Shapes: body0 collision, body0 visual, body1 collision, body2 collision,
    # body3 (world 1) collision, global ground plane.
    shape_body = [0, 0, 1, 2, 3, -1]
    shape_world = [0, 0, 0, 0, 1, -1]
    collide = 2
    flags = [collide, 0, collide, collide, collide, collide]
    rows, columns = shape_axes(
        shape_body,
        shape_world,
        flags,
        collide,
        row_of_body={0: 0, 3: 1},
        column_of_body={1: 1, 3: 2},
        terrain_column=0,
    )
    assert rows.tolist() == [0, -1, -1, -1, 1, -1]
    assert columns.tolist() == [-1, -1, 1, -1, 2, 0]
    # Terrain unselected: the ground shape takes no column at all.
    _, columns = shape_axes(
        shape_body, shape_world, flags, collide, {0: 0}, {1: 0}, terrain_column=None
    )
    assert columns.tolist() == [-1, -1, 0, -1, -1, -1]


def test_contact_force_matrix_projects_normals_handles_both_orders_and_subsets():
    # Shape axes for one env, two rows (R=2), two partners (P=2).
    # Shapes 0,1 are rows; shape 2 is the ground; shape 3 is an unselected body.
    shape_row = torch.tensor([0, 1, -1, -1])
    # Shape 0 is also partner column 1; the ground is column 0.
    shape_column = torch.tensor([1, -1, 0, -1])
    count = torch.tensor([3])
    # k0: row0 vs ground, force on shape0 has a 3 N tangential part -> dropped.
    # k1: ground vs row1 (reversed order), force on ground is -(0,0,20) -> +20 on row1.
    # k2: row0 vs shape3 (unselected partner) -> ignored. Slot 3 is beyond count.
    shape0 = torch.tensor([0, 2, 0, 0])
    shape1 = torch.tensor([2, 1, 3, 1])
    force = torch.tensor(
        [
            [3.0, 0.0, 10.0, 9, 9, 9],
            [0.0, 0.0, -20.0, 0, 0, 0],
            [5.0, 5.0, 5.0, 0, 0, 0],
            [1.0, 1.0, 1.0, 0, 0, 0],
        ]
    )
    normal = torch.tensor(
        [[0.0, 0.0, 2.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    matrix = contact_force_matrix(
        count, shape0, shape1, force, normal, shape_row, shape_column, 1, 2, 2
    )
    expected = torch.zeros(1, 2, 2, 3)
    expected[0, 0, 0, 2] = 10.0  # normal part only; sign of stored normal irrelevant
    expected[0, 1, 0, 2] = 20.0  # reversed ordering flips the force onto row1
    torch.testing.assert_close(matrix, expected)
    # Unused slots carry -1 shape ids and must never be read.
    matrix = contact_force_matrix(
        torch.tensor([0]),
        torch.full((4,), -1),
        torch.full((4,), -1),
        force,
        normal,
        shape_row,
        shape_column,
        1,
        2,
        2,
    )
    torch.testing.assert_close(matrix, torch.zeros(1, 2, 2, 3))


def test_contact_force_matrix_sums_multipoint_pairs_and_keeps_partners_separate():
    shape_row = torch.tensor([0, -1, -1])
    shape_column = torch.tensor([-1, 0, 1])
    shape0 = torch.tensor([0, 0, 0])
    shape1 = torch.tensor([1, 1, 2])
    force = torch.tensor(
        [
            [0.0, 0.0, 10.0, 0, 0, 0],
            [0.0, 0.0, 15.0, 0, 0, 0],
            [-4.0, 0.0, 0.0, 0, 0, 0],
        ]
    )
    normal = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    matrix = contact_force_matrix(
        torch.tensor([3]),
        shape0,
        shape1,
        force,
        normal,
        shape_row,
        shape_column,
        1,
        1,
        2,
    )
    assert matrix[0, 0, 0].tolist() == [0.0, 0.0, 25.0]  # two points, same pair
    assert matrix[0, 0, 1].tolist() == [-4.0, 0.0, 0.0]  # other partner stays separate
