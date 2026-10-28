# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified inference controller: speed limits, diagnostics and recovery."""

import logging
from types import SimpleNamespace

import pytest
import torch

from protomotions.envs.control.masked_mimic_control import (
    MaskedMimicControlConfig,
)
from protomotions.envs.control.masked_mimic_tasks.controller import (
    GetupRecoveryConfig,
    MaskedMimicTaskControl,
    MaskedMimicTaskControlConfig,
)
from protomotions.envs.control.masked_mimic_prompt import GoalCanvas
from protomotions.envs.control.mimic_control import MimicControl
from protomotions.tests.support.task_inputs import (
    make_robot,
    make_robot_state,
    make_task_inputs,
)


@pytest.fixture
def make_controller(monkeypatch):
    monkeypatch.setattr(
        MaskedMimicTaskControl, "_default_upright_height", lambda _: 1.0
    )

    def build(**kwargs):
        pos = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.2]]]
        ).repeat(2, 1, 1)
        rot = torch.zeros(2, 3, 4)
        rot[..., 3] = 1
        vel = torch.zeros_like(pos)
        inputs = make_task_inputs(
            2,
            robot=make_robot(
                ("foot", "anchor", "hand"),
                anchor_body_index=1,
                trackable_bodies_subset=["hand", "anchor"],
                default_root_height=1.0,
                semantic_forward_axis_xy=(1.0, 0.0),
            ),
            dt=0.02,
            headless=False,
        )
        cfg = MaskedMimicTaskControlConfig(
            reach_body_name="hand",
            getup=GetupRecoveryConfig(),
            canvas=GoalCanvas(
                future_steps=(1,),
                num_masked_future_steps=5,
                conditionable_bodies=("hand", "anchor"),
                anchor_body="anchor",
                anchor_pose_submasking=True,
                has_local_stream=True,
                style_frames=None,
                has_velocity_command=True,
            ),
            **kwargs,
        )
        control = MaskedMimicTaskControl(cfg, inputs)
        # The measured state the lifecycle calls take; it shares pos and vel.
        control.test_state = make_robot_state(pos, rotations=rot, velocities=vel)
        control.reset_rows(torch.arange(2), control.test_state)
        return control, pos, vel

    return build


def test_speed_cap_for_initial_scripted_and_keyboard_commands(make_controller):
    control, _, _ = make_controller(
        task="steering", speed=8.0, command_source="keyboard"
    )
    assert (control.speed == 5).all()
    ids = torch.tensor([1])
    control.set_steering(ids, 20.0, 0.0)
    control._adjust_keyboard_speed(ids, 0.1)
    assert control.speed[1] == 5
    control.set_location(ids, torch.tensor([[100.0, 0.0]]), 1.0)
    assert control.deadline[1] == 20
    assert control.location_speed[1] == 5
    control._adjust_keyboard_speed(ids, 20.0)
    assert control.location_speed[1] == 5
    control._adjust_keyboard_speed(ids, -100.0)
    assert control.location_speed[1] == pytest.approx(0.1)
    control.set_steering(ids, 0.0, 0.0)
    control._adjust_keyboard_speed(ids, -0.1)
    assert control.speed[1] == 0


def test_initial_location_and_zero_distance_are_bounded(make_controller):
    control, pos, _ = make_controller(
        task="location", location_speed=10.0, command_source="keyboard"
    )
    assert (control.location_speed == 5).all()
    control._set_keyboard_location(torch.arange(2), pos[:, 1, :2])
    assert (control.location_speed == 5).all()
    assert torch.isfinite(control.deadline).all()
    control, _, _ = make_controller(task="location", location_offset=(100.0, 0.0))
    assert (control.deadline == 20).all()
    assert (control.location_speed == 5).all()


@pytest.mark.parametrize("task", ["idle", "steering", "location", "reaching"])
def test_recovery_masks_all_tasks_and_restores_commands(make_controller, task):
    control, pos, _ = make_controller(task=task)
    original = control.save_state()
    healthy = control._pack_prompt()
    pos[0, 1, 2] = 0.4
    control.advance(control.test_state)
    assert control.recovering.tolist() == [True, False]
    recovering = control._pack_prompt()
    assert not recovering.target_bodies_masks[0].any()
    assert not recovering.target_poses_masks[0].any()
    assert not recovering.velocity_command_mask[0].any()
    assert recovering.local_target_poses_masks[0].all()
    local = recovering.local_target_bodies_masks.view(2, 5, 2, 2)
    assert local[0, :, 1, 0].all()
    assert local[0].sum() == 5  # Only anchor translation, no rotation/hand targets.
    assert not local[1].any()
    torch.testing.assert_close(recovering.local_ref_pos[0, :, 1, 2], torch.ones(5))
    torch.testing.assert_close(recovering.ref_pos[1], healthy.ref_pos[1])
    subset = control._pack_prompt(torch.tensor([0]))
    torch.testing.assert_close(subset.local_ref_pos[0], recovering.local_ref_pos[0])
    pos[0, 1, 2] = 0.8
    control._latch(None, control.test_state)
    control._update_getup(torch.tensor([0]))
    assert control.recovering[0]  # Hysteresis: above entry but below exit.
    pos[0, 1, 2] = 1.0
    control._latch(None, control.test_state)
    control._update_getup(None)
    assert not control.recovering.any()
    resumed = control._pack_prompt()
    torch.testing.assert_close(resumed.target_bodies_masks, healthy.target_bodies_masks)
    for name in ("mode", "speed", "heading", "facing", "position", "reach_position"):
        torch.testing.assert_close(getattr(control, name), original[name])


@pytest.mark.parametrize("task", ["idle", "steering", "location", "reaching"])
def test_periodic_mode_errors_and_recovery_status(make_controller, caplog, task):
    control, pos, vel = make_controller(task=task)
    vel[0, 1, 0] = 0.5
    caplog.set_level(logging.INFO)
    if task == "steering":
        control.set_steering(torch.tensor([0]), 1.0, 0.0, torch.pi / 2)
    control.advance(control.test_state)
    assert task in caplog.text
    if task == "steering":
        assert "velocity error=0.500 m/s" in caplog.text
        assert "heading error=+90.0 deg" in caplog.text
    else:
        assert "location error=" in caplog.text
        assert "speed actual=0.500" in caplog.text
        if task == "reaching":
            assert "reaching error=" in caplog.text
    caplog.clear()
    for _ in range(24):
        control.advance(control.test_state)
    assert not caplog.records
    pos[0, 1, 2] = 0.4
    control.advance(control.test_state)
    assert "recovery; torso height=0.400 m" in caplog.text
    assert "task goals masked" in caplog.text
    caplog.clear()
    control.config.status_print_interval_seconds = 0
    for _ in range(30):
        control.advance(control.test_state)
    assert not caplog.records


def test_status_uses_active_keyboard_robot(make_controller, caplog):
    control, _, vel = make_controller(task="steering", command_source="keyboard")
    control.inputs.ui.active_env_id = 1
    vel[1, 1, 0] = 0.75
    caplog.set_level(logging.INFO)
    control.advance(control.test_state)
    assert "env=1" in caplog.text
    assert "velocity error=0.250 m/s" in caplog.text


def test_inference_task_override_enables_recovery():
    """The example preset derives the canvas from the trained config and the
    robot (a local stream for its get-up recovery, anchor masks bound) and
    turns recovery on."""
    from examples.inference.masked_mimic import apply_inference_overrides
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent

    env = SimpleNamespace(
        control_components={
            "masked_mimic": MaskedMimicControlConfig(
                anchor_pose_submasking=True, local_target_condition_probability=0.5
            )
        },
        observation_components={
            "anchor": MdpComponent(
                lambda m: m, {"m": EnvContext.masked_mimic.anchor_rotation_mask}
            )
        },
        termination_components={},
    )
    robot = SimpleNamespace(
        kinematic_info=SimpleNamespace(body_names=["foot", "anchor", "hand"]),
        anchor_body_index=1,
        trackable_bodies_subset=["hand", "anchor"],
    )
    from protomotions.utils.experiment_configs import ExperimentConfigs

    apply_inference_overrides(
        ExperimentConfigs(
            robot=robot,
            simulator=None,
            terrain=None,
            scene_lib=None,
            motion_lib=None,
            env=env,
            agent=None,
        ),
        SimpleNamespace(full_eval=False),
    )
    task = env.control_components["masked_mimic"]
    assert isinstance(task.getup, GetupRecoveryConfig)
    assert task.canvas.anchor_body == "anchor" and task.canvas.has_local_stream


def test_publish_is_a_pure_read_and_reset_latches_recovery(
    make_controller, monkeypatch
):
    """``publish`` never moves the get-up latch: ``step`` does, for every env,
    and ``reset`` does for the rows it resets, so a robot reset onto the
    ground starts in recovery while the other rows keep their latch."""
    monkeypatch.setattr(MimicControl, "publish", lambda self, env_ids: {})
    control, pos, _ = make_controller(task="idle")
    pos[0, 1, 2] = 0.4  # fallen, but nothing has stepped or reset yet
    before = {
        key: value.clone()
        for key, value in control.save_state().items()
        if isinstance(value, torch.Tensor)
    }
    published = control.publish(None)
    control.publish(torch.tensor([0]))
    assert not control.recovering.any()
    for key, value in before.items():
        assert torch.equal(control.save_state()[key], value), key
    assert "masked_mimic.target_bodies_masks" in published

    control.advance(control.test_state)
    assert control.recovering.tolist() == [True, False]

    pos[0, 1, 2] = 1.0
    pos[1, 1, 2] = 0.4
    control.reset_rows(torch.tensor([1]), control.test_state[torch.tensor([1])])
    assert control.recovering.tolist() == [True, True]  # row 0 was not reset
    control.publish(None)
    assert control.recovering.tolist() == [True, True]
    control.advance(control.test_state)
    assert control.recovering.tolist() == [False, True]
