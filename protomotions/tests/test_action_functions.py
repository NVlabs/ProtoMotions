# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pure action processing functions and config helpers."""

from types import SimpleNamespace

import pytest
import torch

from protomotions.envs.action.action_functions import (
    bm_pd_action,
    build_pd_action_offset_scale,
    make_bm_pd_action_config,
    make_passthrough_pd_action_config,
    make_pd_action_config,
    normalized_pd_fixed_gains_action,
    passthrough_pd_action,
)


def _assert_no_storage_alias(lhs: torch.Tensor, rhs: torch.Tensor):
    assert lhs.data_ptr() != rhs.data_ptr()


def _mixed_dof_robot_config(**control_overrides):
    dof_names = [
        "body3_x",
        "body3_y",
        "body3_z",
        "hinge1",
        "hinge4",
    ]
    default_control = {
        "body3_x": SimpleNamespace(stiffness=10.0, damping=1.0, effort_limit=100.0),
        "body3_y": SimpleNamespace(stiffness=20.0, damping=2.0, effort_limit=80.0),
        "body3_z": SimpleNamespace(stiffness=40.0, damping=4.0, effort_limit=60.0),
        "hinge1": SimpleNamespace(stiffness=50.0, damping=5.0, effort_limit=25.0),
        "hinge4": SimpleNamespace(stiffness=100.0, damping=10.0, effort_limit=10.0),
    }
    default_control.update(control_overrides)
    return SimpleNamespace(
        default_dof_pos=torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        kinematic_info=SimpleNamespace(
            hinge_axes_map={
                3: torch.eye(3),
                1: torch.tensor([[0.0, 0.0, 1.0]]),
                4: torch.tensor([[0.0, 1.0, 0.0]]),
            },
            dof_limits_lower=torch.tensor([-0.3, -0.1, -2.0, -1.0, 0.2]),
            dof_limits_upper=torch.tensor([0.4, 0.5, 1.0, 3.0, 1.2]),
            dof_names=dof_names,
        ),
        control=SimpleNamespace(control_info=default_control),
    )


@pytest.mark.parametrize(
    ("action_transform", "clamp_value", "transformed"),
    [
        ("tanh", 1.0, torch.tanh(torch.tensor([[-2.0, -0.5, 0.5], [0.0, 2.0, 4.0]]))),
        ("clamp", 0.75, torch.tensor([[-0.75, -0.5, 0.5], [0.0, 0.75, 0.75]])),
        (None, 1.0, torch.tensor([[-2.0, -0.5, 0.5], [0.0, 2.0, 4.0]])),
    ],
)
def test_normalized_pd_fixed_gains_action_applies_transform_offset_scale_and_gains(
    action_transform,
    clamp_value,
    transformed,
):
    action = torch.tensor([[-2.0, -0.5, 0.5], [0.0, 2.0, 4.0]])
    pd_action_offset = torch.tensor([0.1, -0.2, 0.3])
    pd_action_scale = torch.tensor([1.0, 2.0, 0.5])
    stiffness = torch.tensor([10.0, 20.0, 30.0])
    damping = torch.tensor([1.0, 2.0, 3.0])

    result = normalized_pd_fixed_gains_action(
        action,
        pd_action_offset,
        pd_action_scale,
        stiffness,
        damping,
        action_transform=action_transform,
        clamp_value=clamp_value,
    )

    assert set(result) == {"processed_action", "stiffness_targets", "damping_targets"}
    assert torch.allclose(
        result["processed_action"], pd_action_offset + pd_action_scale * transformed
    )
    assert torch.equal(result["stiffness_targets"], stiffness.repeat(2, 1))
    assert torch.equal(result["damping_targets"], damping.repeat(2, 1))
    _assert_no_storage_alias(result["processed_action"], action)
    _assert_no_storage_alias(result["stiffness_targets"], stiffness)
    _assert_no_storage_alias(result["damping_targets"], damping)


def test_passthrough_pd_action_clones_action_and_expands_fixed_gains():
    action = torch.tensor([[0.1, -0.2], [0.3, -0.4]])
    stiffness = torch.tensor([12.0, 34.0])
    damping = torch.tensor([1.2, 3.4])

    result = passthrough_pd_action(action, stiffness, damping)

    assert torch.equal(result["processed_action"], action)
    assert torch.equal(result["stiffness_targets"], stiffness.repeat(2, 1))
    assert torch.equal(result["damping_targets"], damping.repeat(2, 1))
    _assert_no_storage_alias(result["processed_action"], action)
    _assert_no_storage_alias(result["stiffness_targets"], stiffness)
    _assert_no_storage_alias(result["damping_targets"], damping)


def test_bm_pd_action_scales_raw_actions_without_bounding_and_clones_outputs():
    action = torch.tensor([[-2.0, 0.5], [3.0, -4.0]])
    pd_action_offset = torch.tensor([0.25, -0.5])
    action_scale = torch.tensor([0.1, 2.0])
    stiffness = torch.tensor([100.0, 200.0])
    damping = torch.tensor([10.0, 20.0])

    result = bm_pd_action(action, pd_action_offset, action_scale, stiffness, damping)

    assert torch.allclose(
        result["processed_action"], torch.tensor([[0.05, 0.5], [0.55, -8.5]])
    )
    assert torch.equal(result["stiffness_targets"], stiffness.repeat(2, 1))
    assert torch.equal(result["damping_targets"], damping.repeat(2, 1))
    _assert_no_storage_alias(result["processed_action"], action)
    _assert_no_storage_alias(result["stiffness_targets"], stiffness)
    _assert_no_storage_alias(result["damping_targets"], damping)


def test_build_pd_action_offset_scale_handles_mixed_hinge_and_three_dof_bodies():
    lower = torch.tensor([-0.3, -0.1, -2.0, -1.0, 0.2])
    upper = torch.tensor([0.4, 0.5, 1.0, 3.0, 1.2])
    hinge_axes_map = {
        3: torch.eye(3),
        1: torch.tensor([[0.0, 0.0, 1.0]]),
        4: torch.tensor([[0.0, 1.0, 0.0]]),
    }

    offset, scale = build_pd_action_offset_scale(
        hinge_axes_map,
        lower,
        upper,
        action_scale=1.5,
        device=torch.device("cpu"),
    )

    assert torch.allclose(offset, torch.tensor([0.05, 0.0, 0.0, 0.0, 0.7]))
    assert torch.allclose(
        scale, torch.tensor([1.05, torch.pi, torch.pi, torch.pi, 1.5])
    )
    assert torch.equal(lower, torch.tensor([-0.3, -0.1, -2.0, -1.0, 0.2]))
    assert torch.equal(upper, torch.tensor([0.4, 0.5, 1.0, 3.0, 1.2]))


def test_build_pd_action_offset_scale_rejects_invalid_dof_size():
    with pytest.raises(ValueError, match="Invalid dof size: 2"):
        build_pd_action_offset_scale(
            {1: torch.eye(2)},
            torch.tensor([-1.0, -1.0]),
            torch.tensor([1.0, 1.0]),
            action_scale=1.0,
            device=torch.device("cpu"),
        )


def test_make_pd_action_config_builds_normalized_action_config_from_robot_data():
    robot_config = _mixed_dof_robot_config()

    config = make_pd_action_config(
        robot_config,
        action_transform="clamp",
        clamp_value=0.2,
        action_scale=1.5,
    )

    assert config["fn"] is normalized_pd_fixed_gains_action
    assert config["action_transform"] == "clamp"
    assert config["clamp_value"] == 0.2
    assert torch.allclose(
        config["pd_action_offset"], torch.tensor([0.05, 0.0, 0.0, 0.0, 0.7])
    )
    assert torch.allclose(
        config["pd_action_scale"],
        torch.tensor([1.05, torch.pi, torch.pi, torch.pi, 1.5]),
    )
    assert torch.equal(
        config["stiffness"], torch.tensor([10.0, 20.0, 40.0, 50.0, 100.0])
    )
    assert torch.equal(config["damping"], torch.tensor([1.0, 2.0, 4.0, 5.0, 10.0]))


def test_make_bm_pd_action_config_uses_default_pose_and_effort_over_stiffness_scale():
    robot_config = _mixed_dof_robot_config()

    config = make_bm_pd_action_config(robot_config)

    assert config["fn"] is bm_pd_action
    assert torch.equal(config["pd_action_offset"], robot_config.default_dof_pos)
    _assert_no_storage_alias(config["pd_action_offset"], robot_config.default_dof_pos)
    assert torch.allclose(
        config["action_scale"], torch.tensor([10.0, 4.0, 1.5, 0.5, 0.1])
    )
    assert torch.equal(
        config["stiffness"], torch.tensor([10.0, 20.0, 40.0, 50.0, 100.0])
    )
    assert torch.equal(config["damping"], torch.tensor([1.0, 2.0, 4.0, 5.0, 10.0]))


def test_make_passthrough_pd_action_config_builds_fixed_gain_config_only():
    robot_config = _mixed_dof_robot_config()

    config = make_passthrough_pd_action_config(robot_config)

    assert config["fn"] is passthrough_pd_action
    assert torch.equal(
        config["stiffness"], torch.tensor([10.0, 20.0, 40.0, 50.0, 100.0])
    )
    assert torch.equal(config["damping"], torch.tensor([1.0, 2.0, 4.0, 5.0, 10.0]))


@pytest.mark.parametrize(
    "config_factory",
    [make_pd_action_config, make_bm_pd_action_config, make_passthrough_pd_action_config],
)
def test_action_config_helpers_reject_missing_joint_control_info(config_factory):
    robot_config = _mixed_dof_robot_config()
    del robot_config.control.control_info["hinge1"]

    with pytest.raises(KeyError, match="hinge1"):
        config_factory(robot_config)


def test_make_bm_pd_action_config_rejects_zero_stiffness_before_dividing():
    robot_config = _mixed_dof_robot_config(
        hinge1=SimpleNamespace(stiffness=0.0, damping=5.0, effort_limit=25.0)
    )

    with pytest.raises(ValueError, match="stiffness must be positive"):
        make_bm_pd_action_config(robot_config)
