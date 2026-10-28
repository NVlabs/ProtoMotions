# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native steering goals: facing/height/velocity commands, deadlines and get-up."""

import math
from types import SimpleNamespace

import pytest
import torch

from protomotions.components.pose_lib import KinematicInfo
from protomotions.envs.control.masked_mimic_tasks.controller import (
    GetupRecoveryConfig,
    MaskedMimicTaskControl,
    MaskedMimicTaskControlConfig,
)
from protomotions.envs.control.masked_mimic_prompt import GoalCanvas
from protomotions.tests.support.task_inputs import make_robot_state, make_task_inputs
from protomotions.utils.robot_heading import semantic_forward_xy

GROUND = 0.25
ROOT_HEIGHT = 0.8
UPRIGHT = 0.5  # torso 0.5 m above the pelvis in the default FK pose


def make_steering(
    forward_axis=(1.0, 0.0), yaw=0.0, num_masked_future_steps=5, **kwargs
):
    """Two standing robots; body order is pelvis, torso (anchor), head."""
    positions = torch.tensor([[[0.0, 0.0, 0.3], [0.0, 0.0, 0.8], [0.0, 0.0, 1.1]]] * 2)
    positions[1, :, :2] += 3.0
    rotation = torch.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])
    rotations = rotation.expand(2, 3, 4).clone()
    kinematic_info = KinematicInfo(
        body_names=["pelvis", "torso", "head"],
        dof_names=["torso_j", "head_j"],
        parent_indices=[-1, 0, 1],
        local_pos=torch.tensor([[0.0] * 3, [0.0, 0.0, 0.5], [0.0, 0.0, 0.3]]),
        local_rot_ref_mat=torch.eye(3).expand(3, 3, 3).clone(),
        hinge_axes_map={
            1: torch.tensor([[0.0, 0.0, 1.0]]),
            2: torch.tensor([[0.0, 0.0, 1.0]]),
        },
        nq=9,
        nv=8,
        num_bodies=3,
        num_dofs=2,
        dof_limits_lower=torch.full((2,), -3.14),
        dof_limits_upper=torch.full((2,), 3.14),
    )
    robot = SimpleNamespace(
        default_root_height=ROOT_HEIGHT,
        default_dof_pos=[0.0, 0.0],
        semantic_forward_axis_xy=forward_axis,
        anchor_body_index=1,
        trackable_bodies_subset=["torso", "head"],
        kinematic_info=kinematic_info,
        mimic_small_marker_bodies=None,
    )
    inputs = make_task_inputs(
        2, robot=robot, dt=0.02, ground_height=GROUND, headless=True
    )
    config = MaskedMimicTaskControlConfig(
        task="steering",
        canvas=GoalCanvas(
            future_steps=(1,),
            num_masked_future_steps=num_masked_future_steps,
            conditionable_bodies=("torso", "head"),
            anchor_body="torso",
            anchor_pose_submasking=True,
            has_local_stream=True,
            style_frames=None,
            has_velocity_command=True,
        ),
        **kwargs,
    )
    control = MaskedMimicTaskControl(config, inputs)
    control.test_state = make_robot_state(positions, rotations=rotations)
    control.reset_rows(torch.arange(2), control.test_state)
    return control, positions


def slots(packed, body=0):
    """Return [E, frames, 2] position/rotation masks for one conditionable body."""
    return packed.target_bodies_masks.view(2, 5, 2, 2)[:, :, body]


def test_standing_steering_goals_match_deployment_layout():
    control, _ = make_steering(speed=0.0, heading=0.0)
    packed = control._pack_prompt()
    anchor = slots(packed)
    expected = torch.zeros(5, 2, dtype=torch.bool)
    expected[0, 1] = True  # facing: rotation in the first slot
    expected[1:3, 0] = True  # height: position in the next two slots
    assert (anchor == expected).all()
    assert not slots(packed, 1).any()
    assert packed.target_poses_masks.tolist() == [[True] * 3 + [False] * 2] * 2
    # Zero turn still shows the minimum horizon; heights use fixed deployment times.
    torch.testing.assert_close(
        packed.time_offsets, torch.tensor([[0.1, 0.5, 1.0, 0.0, 0.0]] * 2)
    )
    torch.testing.assert_close(
        packed.ref_pos[:, 1:3, 1, 2], torch.full((2, 2), GROUND + ROOT_HEIGHT)
    )
    assert packed.anchor_translation_mask[0].tolist() == [
        [False, False],
        [False, True],
        [False, True],
        [False, False],
        [False, False],
    ]
    assert packed.anchor_rotation_mask[0, 0].tolist() == [False, True]
    assert packed.velocity_command_mask.all()
    assert not packed.average_velocity.any()
    assert not packed.local_target_poses_masks.any()


@pytest.mark.parametrize("forward_axis", [(1.0, 0.0), (0.0, 1.0), (0.0, -1.0)])
@pytest.mark.parametrize("facing", [0.0, math.pi / 2, -2.5])
def test_facing_goal_uses_anatomical_forward_axis_and_turn_rate(forward_axis, facing):
    yaw = 0.4
    control, _ = make_steering(forward_axis=forward_axis, yaw=yaw)
    control.set_steering(torch.arange(2), 1.0, 0.3, facing)
    packed = control._pack_prompt()
    goal_forward = semantic_forward_xy(
        packed.ref_rot[:, 0, 1], forward_axis, w_last=True
    )
    torch.testing.assert_close(
        goal_forward,
        torch.tensor([math.cos(facing), math.sin(facing)]).expand(2, 2),
        atol=1e-5,
        rtol=0,
    )
    current = semantic_forward_xy(
        torch.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]),
        forward_axis,
        w_last=True,
    )
    error = facing - math.atan2(current[1], current[0])
    error = (error + math.pi) % (2 * math.pi) - math.pi
    expected = max(abs(error) / math.radians(90.0), 0.1)
    torch.testing.assert_close(packed.time_offsets[:, 0], torch.full((2,), expected))
    # Movement direction is independent of facing and is a world-XY velocity.
    torch.testing.assert_close(
        packed.average_velocity,
        torch.tensor([math.cos(0.3), math.sin(0.3)]).expand(2, 2),
        atol=1e-6,
        rtol=0,
    )


def test_turn_deadline_counts_down_and_keeps_minimum_horizon():
    control, _ = make_steering()
    control.set_steering(torch.arange(2), 0.5, 0.0, math.pi / 2)
    assert control._pack_prompt().time_offsets[0, 0] == pytest.approx(1.0)
    control.inputs.clock.progress_buf[:] = 20  # 0.4 s later
    packed = control._pack_prompt()
    assert packed.time_offsets[0, 0] == pytest.approx(0.6)
    assert packed.time_offsets[0, 1:3].tolist() == pytest.approx([0.5, 1.0])
    control.inputs.clock.progress_buf[:] = 100
    assert control._pack_prompt().time_offsets[0, 0] == pytest.approx(0.1)


def test_default_facing_follows_movement_heading():
    control, _ = make_steering(heading=0.7)
    torch.testing.assert_close(control.facing, torch.full((2,), 0.7))
    control.set_steering(torch.tensor([1]), 1.0, 1.2, facing=-0.4)
    control.set_steering(torch.tensor([0]), 1.0, 1.2)
    assert control.facing.tolist() == pytest.approx([1.2, -0.4])


def fall(positions, torso_height):
    positions[:, 0, 2] = 0.0
    positions[:, 1, 2] = torso_height
    positions[:, 2, 2] = 0.05


def test_getup_replaces_steering_with_repeated_local_height_goals():
    control, positions = make_steering(getup=GetupRecoveryConfig(), speed=1.0)
    assert control.upright_height == pytest.approx(UPRIGHT)
    fall(positions, 0.1)
    control._latch(None, control.test_state)
    control._update_getup(None)
    assert control.recovering.all()
    packed = control._pack_prompt()
    assert not packed.target_poses_masks.any()
    assert not packed.target_bodies_masks.any()
    assert not packed.anchor_translation_mask.any()
    assert not packed.anchor_rotation_mask.any()
    assert not packed.velocity_command_mask.any()
    assert not packed.average_velocity.any()
    assert packed.local_target_poses_masks.all()
    local = packed.local_target_bodies_masks.view(2, 5, 2, 2)
    assert local[:, :, 0, 0].all() and not local[:, :, 1].any()
    assert not local[:, :, 0, 1].any()
    first = (UPRIGHT - 0.1) / 0.8
    expected = (first + torch.arange(5) * 0.25).clamp_max(2.0)
    torch.testing.assert_close(packed.local_time_offsets, expected.expand(2, -1))
    torch.testing.assert_close(packed.local_target_times, expected.expand(2, -1))
    torch.testing.assert_close(
        packed.local_ref_pos[:, :, 1, 2], torch.full((2, 5), UPRIGHT)
    )
    torch.testing.assert_close(
        packed.local_ref_pos[:, :, 0], positions[:, 0, None].expand(-1, 5, -1)
    )


def test_getup_timing_clamps_and_hysteresis():
    control, positions = make_steering(getup=GetupRecoveryConfig())
    torch.testing.assert_close(
        control.getup_time(torch.tensor([0.0, 0.2, 0.6, 1.0, 2.0])),
        torch.tensor([0.25, 0.25, 0.75, 1.25, 1.25]),
    )
    fall(positions, 0.4)  # between enter (0.325) and exit (0.45)
    control._latch(None, control.test_state)
    control._update_getup(None)
    assert not control.recovering.any()
    fall(positions, 0.3)
    control._latch(None, control.test_state)
    control._update_getup(None)
    assert control.recovering.all()
    fall(positions, 0.4)
    control._latch(None, control.test_state)
    control._update_getup(None)
    assert control.recovering.all()
    fall(positions, 0.46)
    control._latch(None, control.test_state)
    control._update_getup(None)
    assert not control.recovering.any()
    assert control._pack_prompt().velocity_command_mask.all()


def test_getup_state_is_per_environment_and_survives_mode_switch():
    control, positions = make_steering(getup=GetupRecoveryConfig(), speed=1.0)
    fall(positions[:1], 0.1)
    control._latch(None, control.test_state)
    control._update_getup(torch.tensor([0]))
    assert control.recovering.tolist() == [True, False]
    packed = control._pack_prompt()
    assert packed.local_target_poses_masks.tolist() == [[True] * 5, [False] * 5]
    assert packed.velocity_command_mask.tolist() == [[False], [True]]
    control.hold(torch.tensor([0]))
    packed = control._pack_prompt()
    assert packed.local_target_poses_masks.tolist() == [[True] * 5, [False] * 5]
    assert packed.velocity_command_mask.tolist() == [[False], [True]]


def test_getup_is_disabled_by_default():
    control, positions = make_steering()
    fall(positions, 0.05)
    control._latch(None, control.test_state)
    control._update_getup(None)
    assert not control.recovering.any()
    assert control._pack_prompt().velocity_command_mask.all()


def test_invalid_steering_and_getup_configs_are_rejected():
    with pytest.raises(ValueError, match="free last frame"):
        make_steering(num_masked_future_steps=3)
    with pytest.raises(ValueError, match="sorted finite height"):
        make_steering(height_target_times_seconds=(1.0, 0.5))
    with pytest.raises(ValueError, match="height fractions"):
        make_steering(getup=GetupRecoveryConfig(enter_height_fraction=0.95))
    with pytest.raises(ValueError, match="time gap <= max time"):
        make_steering(getup=GetupRecoveryConfig(max_time_seconds=0.1))


def test_subset_prompt_matches_full_prompt_for_a_fallen_environment():
    control, positions = make_steering(getup=GetupRecoveryConfig(), speed=1.0)
    fall(positions[1:], 0.1)
    control._latch(None, control.test_state)
    control._update_getup(torch.tensor([1]))
    full = control._pack_prompt()
    subset = control._pack_prompt(torch.tensor([1]))
    torch.testing.assert_close(subset.local_time_offsets[0], full.local_time_offsets[1])
    torch.testing.assert_close(subset.local_ref_pos[0], full.local_ref_pos[1])
    assert not subset.velocity_command_mask.any()
