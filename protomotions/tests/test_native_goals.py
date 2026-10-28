# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native goal packing, stable world targets, switching and shared viewer input."""

from dataclasses import replace
import math
import pickle
from types import SimpleNamespace

import pytest
import torch

from protomotions.envs.control.masked_mimic_tasks.controller import (
    GetupRecoveryConfig,
    MaskedMimicTaskControl,
    MaskedMimicTaskControlConfig,
    configure_masked_mimic_task,
)
from protomotions.envs.control.masked_mimic_prompt import GoalCanvas
from protomotions.envs.control.masked_mimic_tasks.goals import BodyGoal, pack_goals
from protomotions.envs.control.masked_mimic_control import (
    MaskedMimicControlConfig,
    MaskedMimicStyleConfig,
)
from protomotions.envs.context import published_vocabulary_errors
from protomotions.envs.control.mimic_control import MimicControl
from protomotions.envs.control.manager import ControlManager, TeleportQueue
from protomotions.envs.obs.masked_mimic_velocity_command import (
    compute_masked_mimic_velocity_command_obs,
)
from protomotions.utils.config_utils import apply_overrides
from protomotions.utils.experiment_configs import ExperimentConfigs
from protomotions.utils.hydra_replacement import get_class
from protomotions.utils import rotations
from protomotions.tests.support.task_inputs import (
    make_robot,
    make_robot_state,
    make_task_inputs,
)


def test_task_config_roundtrip_resolves_concrete_controller():
    config = MaskedMimicTaskControlConfig(canvas=canvas(), getup=GetupRecoveryConfig())
    restored = pickle.loads(pickle.dumps(config))
    assert restored == config
    assert get_class(restored._target_) is MaskedMimicTaskControl
    assert type(restored).__module__.endswith("masked_mimic_tasks.controller")


def canvas(num_masked_future_steps=5, velocity=True, style_frames=5, local=True):
    """The trained layout of the two-body fake robot below (anchor "anchor")."""
    return GoalCanvas(
        future_steps=(1,),
        num_masked_future_steps=num_masked_future_steps,
        conditionable_bodies=("hand", "anchor"),
        anchor_body="anchor",
        anchor_pose_submasking=True,
        has_local_stream=local,
        style_frames=style_frames,
        has_velocity_command=velocity,
    )


def make_control(*, num_masked_future_steps=5, velocity=True, headless=False, **kwargs):
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 2.0, 0.8], [1.4, 2.0, 1.2]],
            [[3.0, 0.0, 0.0], [4.0, 2.0, 0.8], [4.4, 2.0, 1.2]],
        ]
    )
    rotations = torch.zeros(2, 3, 4)
    rotations[..., 3] = 1
    # The ground query a test may swap (``control.test_ground.fn``).
    ground = SimpleNamespace(fn=lambda locations: torch.zeros(len(locations)))
    inputs = make_task_inputs(
        2,
        robot=make_robot(
            ("base", "anchor", "hand"),
            anchor_body_index=1,
            trackable_bodies_subset=["hand", "anchor"],
            default_root_height=0.8,
        ),
        dt=0.02,
        ground_height=lambda locations: ground.fn(locations).reshape(-1),
        headless=headless,
    )
    config_values = dict(
        reach_body_name="hand",
        canvas=canvas(num_masked_future_steps=num_masked_future_steps, velocity=velocity),
    )
    config_values.update(kwargs)
    cfg = MaskedMimicTaskControlConfig(**config_values)
    control = MaskedMimicTaskControl(cfg, inputs)
    # The measured state the lifecycle calls take; it shares ``positions``.
    control.test_state = make_robot_state(positions, rotations=rotations)
    control.test_ground = ground
    control.reset_rows(torch.arange(2), control.test_state)
    return control, positions


def test_packer_uses_canvas_order_nonroot_anchor_and_deadlines():
    clock = torch.tensor([1.0, 2.0])
    position = torch.tensor([[5.0, 6.0, 1.0], [float("nan"), 0.0, 0.0]])
    goal = BodyGoal(
        2,
        1,
        torch.tensor([3.0, float("nan")]),
        torch.tensor([True, False]),
        position=position,
    )
    output = pack_goals(
        [goal],
        clock=clock,
        body_count=3,
        conditionable_ids=[2, 1],
        anchor_id=1,
        frames=3,
        style_frames=2,
    )
    assert output.target_bodies_masks.view(2, 3, 2, 2)[0, 1, 0, 0]
    assert output.target_poses_masks.tolist() == [
        [False, True, False],
        [False, False, False],
    ]
    torch.testing.assert_close(output.ref_pos[0, 1, 2], position[0])
    assert output.time_offsets.tolist() == [[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]
    assert torch.isfinite(output.ref_pos).all()
    assert not output.local_target_bodies_masks.any()
    assert not output.style_pose_samples.any()
    assert not output.velocity_command_mask.any()


def test_packer_rejects_conflicting_frame_times_and_duplicate_bodies():
    clock = torch.zeros(2)
    goal = BodyGoal(
        2, 0, clock + 2, torch.ones(2, dtype=torch.bool), position=torch.zeros(2, 3)
    )
    args = dict(
        clock=clock,
        body_count=3,
        conditionable_ids=[1, 2],
        anchor_id=1,
        frames=3,
        style_frames=0,
    )
    with pytest.raises(ValueError, match="once"):
        pack_goals([goal, goal], **args)
    with pytest.raises(ValueError, match="deadline"):
        pack_goals([goal, replace(goal, body_id=1, deadline=clock + 3)], **args)
    with pytest.raises(ValueError, match="unit"):
        pack_goals([replace(goal, rotation=torch.zeros(2, 4))], **args)


@pytest.mark.parametrize("task", ["idle", "location", "reaching"])
@pytest.mark.parametrize("frames", [1, 2, 3])
def test_nonsteering_tasks_keep_their_original_canvas(task, frames):
    control, _ = make_control(task=task, num_masked_future_steps=frames)
    packed = control._pack_prompt()
    assert packed.ref_pos.shape[1] == frames
    assert packed.target_poses_masks[:, -1].all()
    assert not packed.velocity_command_mask.any()


@pytest.mark.parametrize("task", ["idle", "location"])
@pytest.mark.parametrize("frames", [1, 2, 3])
def test_scripted_steering_rejects_small_canvas_before_changing_state(task, frames):
    control, _ = make_control(task=task, num_masked_future_steps=frames)
    before = {name: tensor.clone() for name, tensor in control.state.tensors()}

    with pytest.raises(ValueError, match="free last frame"):
        control.set_steering(torch.tensor([0]), speed=1.0, heading=0.5)

    for name, value in before.items():
        torch.testing.assert_close(dict(control.state.tensors())[name], value)
    control._pack_prompt()  # The original non-steering command still works.


@pytest.mark.parametrize("frames", [4, 5])
def test_scripted_steering_switch_packs_a_valid_canvas(frames):
    control, _ = make_control(task="location", num_masked_future_steps=frames)
    control.set_steering(torch.tensor([0]), speed=1.0, heading=0.5, facing=0.1)

    packed = control._pack_prompt()
    assert control.mode.tolist() == [control.MODES["steering"], control.MODES["location"]]
    assert packed.velocity_command_mask.reshape(-1).tolist() == [True, False]
    assert packed.ref_pos.shape[1] == frames


@pytest.mark.parametrize("kwargs", [
    {"turn_rate_degrees_per_second": 0.0},
    {"min_target_time_seconds": float("nan")},
    {"height_target_times_seconds": (1.0, 0.5)},
    {"height_target_times_seconds": (0.5, float("inf"))},
])
def test_scripted_steering_checks_timing_before_switching(kwargs):
    control, _ = make_control(task="idle", **kwargs)
    before = {name: tensor.clone() for name, tensor in control.state.tensors()}

    with pytest.raises(ValueError, match="Steering needs"):
        control.set_steering(torch.tensor([0]), speed=1.0, heading=0.5)

    for name, value in before.items():
        torch.testing.assert_close(dict(control.state.tensors())[name], value)


def test_world_targets_do_not_drift_and_deadlines_only_change_on_command():
    control, positions = make_control()
    control.set_location(torch.tensor([0]), torch.tensor([[8.0, 9.0]]), 2.0)
    original = control._pack_prompt()
    positions[0] += 50
    control.inputs.clock.progress_buf[:] = 25
    control.advance(control.test_state)
    later = control._pack_prompt()
    torch.testing.assert_close(original.ref_pos, later.ref_pos)
    torch.testing.assert_close(original.target_times[:, -1], later.target_times[:, -1])
    assert later.time_offsets[0, -1] == 1.5
    control.inputs.clock.progress_buf[:] = 150
    assert control._pack_prompt().time_offsets[0, -1] == 0
    # An inference task owns no episode-ending check.
    robot = control.inputs.robot
    assert MaskedMimicTaskControl.terminations(control.config, robot) == {}


def test_target_grounding_is_invariant_to_world_height_translation():
    from protomotions.envs.obs.masked_mimic import _grounded_body_positions

    control, _ = make_control()
    control.set_reaching(
        torch.arange(2), torch.tensor([[2.0, 3.0, 1.2], [4.0, 5.0, 1.2]]), 2.0
    )
    expected = _grounded_body_positions(control._pack_prompt().ref_pos)
    control.test_ground.fn = lambda locations: torch.full(
        (len(locations),), 10.0
    )
    shifted = _grounded_body_positions(control._pack_prompt().ref_pos)
    torch.testing.assert_close(shifted, expected)


def test_goal_heights_follow_terrain_under_the_goal():
    from protomotions.envs.obs.masked_mimic import _grounded_body_positions

    control, positions = make_control()
    control.test_ground.fn = lambda locations: torch.where(
        locations[:, 0] > 6.0, 3.0, 0.0
    )
    anchor_height = positions[:, 1, 2]
    control.set_location(torch.arange(2), torch.tensor([[8.0, 1.0], [8.0, 2.0]]), 2.0)
    grounded = _grounded_body_positions(control._pack_prompt().ref_pos)
    torch.testing.assert_close(grounded[:, -1, 1, 2], anchor_height)
    # Reaching re-anchors at the robot (flat ground); the hand keeps its world
    # height and the pose is grounded under the anchor, not under the hand.
    control.set_reaching(
        torch.arange(2), torch.tensor([[8.0, 1.0, 4.2], [8.0, 2.0, 4.2]]), 2.0
    )
    packed = control._pack_prompt()
    torch.testing.assert_close(packed.ref_pos[:, -1, 2, 2], torch.tensor([4.2, 4.2]))
    torch.testing.assert_close(packed.ref_pos[:, -1, 0, 2], torch.zeros(2))
    grounded = _grounded_body_positions(packed.ref_pos)
    torch.testing.assert_close(grounded[:, -1, 2, 2], torch.tensor([4.2, 4.2]))
    torch.testing.assert_close(grounded[:, -1, 1, 2], anchor_height)


def test_switching_clears_old_modalities_and_velocity_is_world_xy():
    control, _ = make_control()
    ids = torch.tensor([0])
    control.set_reaching(ids, torch.tensor([[2.0, 3.0, 1.0]]), 2.0)
    reaching = control._pack_prompt()
    assert reaching.target_bodies_masks.view(2, 5, 2, 2)[0, -1, 0, 0]
    control.set_steering(ids, 2.0, torch.pi / 2)
    steering = control._pack_prompt()
    masks = steering.target_bodies_masks.view(2, 5, 2, 2)
    assert masks[0].sum() == 3
    assert masks[0, 0, 1, 1] and masks[0, 1:3, 1, 0].all()
    assert not masks[0, -1].any()
    assert not steering.anchor_translation_mask[0, 0].any()
    assert steering.velocity_command_mask.tolist() == [[True], [False]]
    torch.testing.assert_close(
        steering.average_velocity[0], torch.tensor([0.0, 2.0]), atol=1e-6, rtol=1e-6
    )
    yaw = torch.tensor([[0.0, 0.0, 2**-0.5, 2**-0.5]]).expand(2, -1)
    obs = compute_masked_mimic_velocity_command_obs(
        yaw, steering.average_velocity, steering.velocity_command_mask
    )
    torch.testing.assert_close(obs[0], torch.tensor([2.0, 0.0]), atol=1e-6, rtol=1e-6)
    assert not obs[1].any()
    control.hold(ids)
    idle = control._pack_prompt()
    assert not idle.velocity_command_mask.any()
    assert not idle.style_frame_masks.any()
    assert not idle.local_target_poses_masks.any()


def test_subset_reset_and_save_restore_do_not_change_other_environments():
    control, positions = make_control()
    control.set_steering(torch.tensor([1]), 1.5, 0.7)
    saved = control.save_state()
    positions[0] += 4
    control.inputs.clock.progress_buf[:] = 100
    control.reset_rows(torch.tensor([0]), control.test_state[torch.tensor([0])])
    control.inputs.clock.progress_buf[0] = 0
    assert control.deadline[0] == control.config.horizon_seconds
    assert control.mode[1] == control.MODES["steering"]
    torch.testing.assert_close(control.position[1], saved["position"][1])
    control.load_state(saved)
    for key, value in saved.items():
        torch.testing.assert_close(dict(control.state.tensors())[key], value)
    subset = control._pack_prompt(torch.tensor([1]))
    full = control._pack_prompt()
    torch.testing.assert_close(subset.ref_pos[0], full.ref_pos[1])


def test_manager_restores_command_state_after_temporary_reset():
    control, _ = make_control()
    manager = ControlManager({}, control.inputs, TeleportQueue())
    manager.components["masked_mimic"] = control
    ids = torch.tensor([1])
    control.set_steering(ids, 1.5, 0.7)
    snapshot = manager.save_state()
    control.reset_rows(ids, control.test_state[ids])
    assert control.mode[1] == control.MODES["idle"]
    manager.load_state(snapshot)
    assert control.mode[1] == control.MODES["steering"]
    torch.testing.assert_close(control.speed[ids], torch.tensor([1.5]))
    with pytest.raises(KeyError, match="masked_mimic"):
        manager.load_state({"wrong_component": snapshot["masked_mimic"]})


def test_context_and_markers_share_same_world_goals(monkeypatch):
    control, _ = make_control()
    control.set_location(torch.tensor([0]), torch.tensor([[8.0, 9.0]]), 2.0)
    mimic = {"mimic.ref_anchor_pos": torch.tensor([[1.0, 2.0, 0.8], [4.0, 2.0, 0.8]])}
    monkeypatch.setattr(MimicControl, "publish", lambda _, env_ids: dict(mimic))
    ctx = control.publish(None)
    markers = control.get_markers_state(control.test_state)["native_goals"]
    torch.testing.assert_close(
        markers.translation[0, 1], ctx["masked_mimic.ref_pos"][0, -1, 1]
    )
    # The packed goals publish exactly the declared masked_mimic.* names.
    goals = {
        key: value for key, value in ctx.items() if key.startswith("masked_mimic.")
    }
    assert published_vocabulary_errors(goals) == []
    assert (markers.translation[:, 0, 2] == -1000).all()
    headless, _ = make_control(headless=True)
    assert headless.get_markers_state(headless.test_state) == {}
    assert control.create_visualization_markers(True) == {}


def test_keyboard_uses_active_environment_and_unregisters():
    control, _ = make_control(task="steering", command_source="keyboard")
    ui = control.inputs.ui
    ui.active_env_id = 1
    ui.handle_key_press("W")
    control.advance(control.test_state)
    torch.testing.assert_close(control.speed, torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(control.heading, torch.tensor([0.0, math.radians(10)]))
    torch.testing.assert_close(control.facing, torch.zeros(2))
    ui.handle_key_press("A")
    control.advance(control.test_state)
    torch.testing.assert_close(control.heading, torch.tensor([0.0, math.radians(10)]))
    torch.testing.assert_close(control.facing, torch.tensor([0.0, math.radians(10)]))
    ui.handle_key_press("S")
    control.advance(control.test_state)
    torch.testing.assert_close(control.heading, torch.zeros(2))
    torch.testing.assert_close(control.facing, torch.tensor([0.0, math.radians(10)]))
    ui.handle_key_press("D")
    control.advance(control.test_state)
    torch.testing.assert_close(control.facing, torch.zeros(2))
    torch.testing.assert_close(control.speed, torch.ones(2))
    control.close()
    assert "native_goals" not in ui.help_text()


def test_keyboard_leaves_simulator_bindings_available():
    control, _ = make_control(command_source="keyboard")
    ui = control.inputs.ui
    # (R is the env's own reset key, registered by make_task_inputs as the env does.)
    for key in ("Q", "J", "L", "O"):
        ui.register_key(key, owner="simulator", description="Existing viewer command")
    control.close()


def test_keyboard_cycles_modes_and_controls_only_selected_environment():
    control, positions = make_control(command_source="keyboard")
    ui = control.inputs.ui
    ui.active_env_id = 1
    control.inputs.clock.progress_buf[:] = 100

    def press(key):
        ui.handle_key_press(key)
        control.advance(control.test_state)

    for key in "WASDUI":
        press(key)
    assert control.mode.tolist() == [0, 0]
    torch.testing.assert_close(control.position, positions[:, 1])

    press("X")
    assert control.mode.tolist() == [0, control.MODES["location"]]
    press("W")
    press("A")
    torch.testing.assert_close(control.position[1, :2], torch.tensor([4.2, 2.2]))
    press("U")
    assert control.position[1, 2] == positions[1, 1, 2]

    press("X")
    assert control.mode.tolist() == [0, control.MODES["steering"]]
    assert control.speed[1] == 0
    assert control.heading[1] == 0
    press("W")
    press("A")
    assert control.speed[1] == 0
    torch.testing.assert_close(control.heading[1], torch.tensor(math.radians(10)))
    torch.testing.assert_close(control.facing[1], torch.tensor(math.radians(10)))
    deadline = control.deadline.clone()
    press("I")
    torch.testing.assert_close(control.deadline, deadline)

    press("X")
    assert control.mode.tolist() == [0, control.MODES["reaching"]]
    torch.testing.assert_close(control.reach_position[1], positions[1, 2])
    press("D")
    press("U")
    torch.testing.assert_close(control.reach_position[1], torch.tensor([4.4, 1.8, 1.3]))

    press("X")
    assert control.mode.tolist() == [0, 0]
    torch.testing.assert_close(control.position, positions[:, 1])
    assert control.inputs.clock.progress_buf.tolist() == [100, 100]


@pytest.mark.parametrize(
    "velocity,reach,expected",
    [(False, False, [2, 0]), (True, False, [2, 3, 0]), (False, True, [2, 1, 0])],
)
def test_keyboard_skips_unconfigured_modes(velocity, reach, expected):
    control, _ = make_control(
        command_source="keyboard",
        velocity=velocity,
        reach_body_name="hand" if reach else None,
    )
    ui = control.inputs.ui
    for mode in expected:
        ui.handle_key_press("X")
        control.advance(control.test_state)
        assert control.mode[0] == mode
        assert control.mode[1] == 0


def test_keyboard_validates_steering_canvas_even_when_starting_idle():
    with pytest.raises(ValueError, match="Steering needs"):
        make_control(command_source="keyboard", num_masked_future_steps=3)


def test_keyboard_enters_steering_facing_the_robots_semantic_forward():
    control, _ = make_control(task="location", command_source="keyboard")
    control.forward_axis = (0.0, 1.0)
    control.inputs.ui.handle_key_press("X")
    control.advance(control.test_state)
    assert control.mode[0] == control.MODES["steering"]
    assert control.speed[0] == 0
    torch.testing.assert_close(control.facing[0], torch.tensor(torch.pi / 2))
    torch.testing.assert_close(control.heading[0], control.facing[0])


@pytest.mark.parametrize("task", ["steering", "location", "idle", "reaching"])
def test_keyboard_speed_keys_are_mode_local_and_per_environment(task):
    control, _ = make_control(task=task, command_source="keyboard")
    ui = control.inputs.ui
    ui.active_env_id = 1
    before = control.save_state()
    ui.handle_key_press("C")
    control.advance(control.test_state)
    if task == "steering":
        torch.testing.assert_close(control.speed, torch.tensor([1.0, 1.1]))
    elif task == "location":
        torch.testing.assert_close(control.location_speed, torch.tensor([1.5, 1.6]))
        torch.testing.assert_close(control.deadline[1], torch.tensor(2.0 / 1.6))
        assert control.deadline[1] < before["deadline"][1]
        torch.testing.assert_close(control.position, before["position"])
    else:
        for name, value in before.items():
            torch.testing.assert_close(getattr(control, name), value)
    for _ in range(30):
        ui.handle_key_press("Z")
        control.advance(control.test_state)
    assert control.speed[1] >= 0
    assert control.location_speed[1] >= 0.1
    for name, value in before.items():
        torch.testing.assert_close(getattr(control, name)[0], value[0])


def test_keyboard_location_movement_uses_selected_speed_and_restores_it():
    control, _ = make_control(task="location", command_source="keyboard")
    ui = control.inputs.ui
    ui.active_env_id = 1
    control.location_speed[1] = 2.0
    control.inputs.clock.progress_buf[:] = 50
    ui.handle_key_press("W")
    control.advance(control.test_state)
    torch.testing.assert_close(control.deadline[1], torch.tensor(1.0 + 2.2 / 2.0))
    saved = control.save_state()
    control.reset_rows(torch.tensor([1]), control.test_state[torch.tensor([1])])
    assert control.location_speed[1] == 1.5
    control.load_state(saved)
    assert control.location_speed[1] == 2.0


def test_steering_arrows_show_world_headings_and_hide_outside_steering():
    control, positions = make_control()
    control.set_steering(torch.tensor([1]), 0.0, 0.0, torch.pi / 2)
    configs = control.create_visualization_markers(False)
    assert all(marker.size == "regular" for marker in configs["native_goals"].markers)
    state = control.get_markers_state(control.test_state)
    for name, direction in (
        ("native_movement", [1.0, 0.0, 0.0]),
        ("native_facing", [0.0, 1.0, 0.0]),
    ):
        assert configs[name].type == "arrow"
        marker = state[name]
        assert marker.translation[0, 0, 2] == -1000
        assert marker.translation[1, 0, 2] == positions[1, 0, 2]
        forward = rotations.quat_rotate(
            marker.orientation[1], torch.tensor([[1.0, 0.0, 0.0]]), True
        )
        torch.testing.assert_close(
            forward[0], torch.tensor(direction), atol=1e-6, rtol=0
        )
    control.set_steering(torch.tensor([1]), 0.0, 0.0)
    state = control.get_markers_state(control.test_state)
    torch.testing.assert_close(
        state["native_movement"].translation,
        state["native_facing"].translation,
    )
    torch.testing.assert_close(
        state["native_movement"].orientation,
        state["native_facing"].orientation,
    )
    control.hold(torch.tensor([1]))
    state = control.get_markers_state(control.test_state)
    for name in ("native_movement", "native_facing"):
        assert (state[name].translation[:, 0, 2] == -1000).all()


def _robot_with_bodies(body_names, anchor_body_index, trackable):
    return SimpleNamespace(
        kinematic_info=SimpleNamespace(body_names=list(body_names)),
        anchor_body_index=anchor_body_index,
        trackable_bodies_subset=list(trackable),
    )


def test_config_adapter_preserves_dimensions_and_rejects_wrong_contract():
    """configure_masked_mimic_task carries the trained layout as one canvas
    (derived from the training config and the robot, checked against the
    bound observations) and refuses a checkpoint without anchor sub-masking."""
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent

    previous = MaskedMimicControlConfig(
        num_masked_future_steps=7,
        future_steps=[1, 3],
        anchor_pose_submasking=True,
        style=MaskedMimicStyleConfig(
            num_frames=3, visible_frame_count_probabilities=(0.25,) * 4
        ),
    )
    robot = _robot_with_bodies(["base", "anchor", "hand"], 1, ["hand", "anchor"])
    env = SimpleNamespace(
        control_components={"masked_mimic": previous},
        termination_components={"old": 1},
        observation_components={
            "anchor": MdpComponent(
                lambda m: m, {"m": EnvContext.masked_mimic.anchor_rotation_mask}
            )
        },
    )
    configure_masked_mimic_task(env, robot, "location")
    new = env.control_components["masked_mimic"]
    assert new.canvas.num_masked_future_steps == 7
    assert new.canvas.style_frames == 3
    assert new.canvas.conditionable_bodies == ("hand", "anchor")
    assert new.canvas.anchor_body == "anchor"
    assert new.future_steps == (1, 3)
    assert new.reference_height_smoothing_seconds == 0
    assert env.termination_components == {}
    assert new.task == "location"
    with pytest.raises(ValueError):
        apply_overrides(
            ExperimentConfigs(None, None, None, None, None, env, None),
            {"env.control_components.masked_mimic.visible_target_pose_prob": 0.1},
        )
    with pytest.raises(ValueError, match="exactly one"):
        configure_masked_mimic_task(SimpleNamespace(control_components={}), robot, "idle")
    with pytest.raises(ValueError, match="submode"):
        configure_masked_mimic_task(
            SimpleNamespace(control_components={"x": MaskedMimicControlConfig()}),
            robot,
            "idle",
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon_seconds": 0},
        {"speed": -1},
        {"heading": float("nan")},
        {"command_source": "random"},
    ],
)
def test_invalid_commands_fail_before_state_changes(kwargs):
    with pytest.raises(ValueError):
        make_control(**kwargs)
    control, _ = make_control()
    saved = control.save_state()
    with pytest.raises(ValueError):
        control.set_location(
            torch.tensor([0]), torch.tensor([[float("nan"), 0.0]]), 1.0
        )
    for key, value in saved.items():
        torch.testing.assert_close(getattr(control, key), value)
