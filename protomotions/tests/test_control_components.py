# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Control-component lifecycle tests on the one shared fake (``make_task_inputs``).

Every control is built from its config and a ``TaskInputs``; the robot state
the lifecycle calls take is built by ``make_robot_state``; a control's
episode-ending checks are read back as ``TerminationRule`` entries and run
over a hand-built context.
"""

import json
from types import SimpleNamespace

import pytest
import torch

from protomotions.envs.context import (
    FrozenContext,
    published_vocabulary_errors,
    requirement_errors,
)
from protomotions.envs.context_requirements import Facts, status
from protomotions.envs.context_views import declared_paths
from protomotions.envs.control.external_kinematic_control import (
    ExternalKinematicControl,
    ExternalKinematicControlConfig,
)
from protomotions.envs.control.gpc_sft_reference_target_control import (
    GPCSFTReferenceTargetControl,
    GPCSFTReferenceTargetControlConfig,
)
from protomotions.envs.control.kinematic_replay_control import (
    KinematicReplayControl,
    KinematicReplayControlConfig,
)
from protomotions.envs.control.masked_mimic_control import (
    FixedBodyCondition,
    MaskedMimicControl,
    MaskedMimicControlConfig,
    MaskedMimicStyleConfig,
    MaskedMimicVelocityCommandConfig,
)
from protomotions.envs.control.path_follower_control import (
    PathFollowerControl,
    PathFollowerControlConfig,
)
from protomotions.envs.control.steering_control import (
    SteeringControl,
    SteeringControlConfig,
)
from protomotions.envs.control import command_source, target_control
from protomotions.envs.control.command_source import (
    KeyboardTargetCommandSourceConfig,
    RandomTargetCommandSourceConfig,
    TargetKeyBindingConfig,
)
from protomotions.envs.control.target_control import (
    TargetControl,
    TargetControlConfig,
)
from protomotions.envs.motion_manager.motion_manager import MotionManager
from protomotions.simulator.base_simulator.contacts import ContactLayout
from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    StateConversion,
)
from protomotions.tests.support.fake_reference import fake_reference
from protomotions.utils.config_schema import from_tree, to_tree
from protomotions.tests.support.task_inputs import (
    identity_quat,
    make_object_state,
    make_reset_state,
    make_robot,
    make_robot_state,
    make_task_inputs,
)


def _flags(rules, published, **extra):
    """Run every rule over a context of the control's published names plus
    ``extra`` top-level names; the flag is the rule's cut of its kernel."""
    ctx = FrozenContext({**published, **extra}, env_ids=None, dt=0.1)
    flags = {}
    for name, rule in rules.items():
        value = rule.component.compute(ctx)
        if rule.threshold is None:
            flags[name] = value.bool()
        elif rule.fail_above:
            flags[name] = value > rule.threshold
        else:
            flags[name] = value < rule.threshold
    return flags


# ---------------------------------------------------------------------------
# Target control
# ---------------------------------------------------------------------------


def test_old_command_source_path_is_refused_not_re_exported():
    """A bundle written before the command sources moved names them under
    ``target_control``. The old path must be refused with the regenerate
    message, which holds only while ``target_control.py`` imports the module
    and not the names: a name imported there is an attribute of the old
    module too, and the bundle would load and die at env build instead.
    """
    for name in command_source.__all__:
        assert not hasattr(target_control, name), name
    tree = to_tree(TargetControlConfig(), where="control")
    old = json.loads(
        json.dumps(tree).replace(
            "protomotions.envs.control.command_source.",
            "protomotions.envs.control.target_control.",
        )
    )
    assert old["command_source"]["$type"].startswith(
        "protomotions.envs.control.target_control."
    )
    with pytest.raises(ValueError, match="renamed or removed; regenerate"):
        from_tree(old, None, where="control", robot=None)


def test_target_control_fixed_target_lifecycle_context_and_markers():
    inputs = make_task_inputs(2, dt=0.1, ground_height=0.25, headless=False)
    inputs.clock.progress_buf[:] = torch.tensor([5, 6])
    config = TargetControlConfig(
        command_source=RandomTargetCommandSourceConfig(
            fixed_target_position=(2.5, -1.5),
        ),
        tar_proximity_threshold=0.4,
        enable_fall_termination=False,
    )
    control = TargetControl(config, inputs)
    state = make_robot_state([[1.0, 2.0, 0.4], [3.0, 4.0, 0.8]])

    control.reset_rows(torch.tensor([0, 1]), state)
    published = control.publish(None)
    markers = control.get_markers_state(state)

    assert torch.allclose(
        published["target.tar_pos"],
        torch.tensor([[2.5, -1.5, 0.25], [2.5, -1.5, 0.25]]),
    )
    assert published["target.tar_proximity_threshold"] == 0.4
    # No check is on: the three check names stay unpublished and no rule exists.
    assert "target.fall_ref_height" not in published
    assert published_vocabulary_errors(published) == []
    assert TargetControl.terminations(config, inputs.robot) == {}
    assert torch.isinf(control.tar_change_time).all()
    assert control.create_visualization_markers(headless=True) == {}
    assert (
        len(
            control.create_visualization_markers(headless=False)[
                "target_markers"
            ].markers
        )
        == 1
    )
    assert torch.allclose(
        markers["target_markers"].translation[:, 0, 2], torch.tensor([0.35, 0.35])
    )

    subset = control.publish(torch.tensor([1]))
    assert torch.equal(subset["target.tar_pos"], control.tar_pos[[1]])

    headless = TargetControl(config, make_task_inputs(2, dt=0.1, headless=True))
    assert headless.get_markers_state(state) == {}


def test_target_control_uses_nonzero_anchor_for_support_state():
    inputs = make_task_inputs(
        2, robot=make_robot(("root", "pelvis"), anchor_body_index=1), dt=0.1
    )
    control = TargetControl(
        TargetControlConfig(
            command_source=RandomTargetCommandSourceConfig(
                fixed_target_position=(2.0, 3.0)
            ),
            enable_fall_termination=False,
        ),
        inputs,
    )
    state = make_robot_state([[0.0, 0.0, 0.1], [1.0, 0.0, 0.2]], num_bodies=2)
    state.rigid_body_pos[:, 1] = torch.tensor([[5.0, 0.0, 1.1], [6.0, 0.0, 1.2]])

    control.reset_rows(torch.tensor([0, 1]), state)

    assert torch.equal(control.last_support_root_height, torch.tensor([1.1, 1.2]))


def test_target_control_resamples_on_schedule_and_clamps_bounds():
    torch.manual_seed(5)
    inputs = make_task_inputs(2, dt=0.5, ground_height=0.4)
    inputs.clock.progress_buf[:] = torch.tensor([4, 0])
    control = TargetControl(
        TargetControlConfig(
            command_source=RandomTargetCommandSourceConfig(
                tar_change_time_min=1.0,
                tar_change_time_max=1.0,
                tar_dist_max=100.0,
                target_bounds=(-1.0, 1.0, -2.0, 2.0),
            ),
            enable_fall_termination=False,
        ),
        inputs,
    )
    control.tar_change_time[:] = torch.tensor([1.0, 100.0])
    control.tar_pos[:] = torch.tensor([[9.0, 9.0, 9.0], [0.25, 0.5, 0.4]])

    control.advance(make_robot_state([[4.0, -4.0, 1.0], [0.0, 0.0, 1.0]]))
    changed_target = control.tar_pos[0].clone()
    unchanged_target = control.tar_pos[1].clone()

    assert -1.0 <= changed_target[0] <= 1.0
    assert -2.0 <= changed_target[1] <= 2.0
    assert changed_target[2] == 0.4
    assert torch.allclose(unchanged_target, torch.tensor([0.25, 0.5, 0.4]))
    assert torch.allclose(control.tar_change_time, torch.tensor([3.0, 100.0]))


def test_target_control_keyboard_source_uses_registered_keys_for_active_env_only():
    inputs = make_task_inputs(2, dt=0.1, headless=False)
    control = TargetControl(
        TargetControlConfig(
            command_source=KeyboardTargetCommandSourceConfig(
                key_bindings=(
                    TargetKeyBindingConfig(
                        key="W",
                        action="move_forward",
                        delta_xy=(0.0, 1.0),
                        description="Move target forward",
                    ),
                ),
                fail_if_headless=False,
            ),
            enable_fall_termination=False,
        ),
        inputs,
    )
    control.tar_pos[:, :2] = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    inputs.ui.active_env_id = 1

    inputs.ui.handle_key_event("W", pressed=True)
    control.advance(make_robot_state([[0.0, 0.0, 1.0], [10.0, 0.0, 1.0]]))

    assert torch.allclose(
        control.tar_pos[:, :2], torch.tensor([[1.0, 1.0], [2.0, 3.0]])
    )


def test_target_control_keyboard_source_rejects_headless_simulator_by_default():
    with pytest.raises(RuntimeError, match="headless"):
        TargetControl(
            TargetControlConfig(
                command_source=KeyboardTargetCommandSourceConfig(
                    key_bindings=(
                        TargetKeyBindingConfig(
                            key="W",
                            action="move_forward",
                            delta_xy=(0.0, 1.0),
                            description="Move target forward",
                        ),
                    )
                ),
                enable_fall_termination=False,
            ),
            make_task_inputs(1, dt=0.1, headless=True),
        )


def test_target_control_gap_and_stuck_checks_respect_grace_period():
    inputs = make_task_inputs(2, dt=0.1)
    inputs.clock.progress_buf[:] = torch.tensor([3, 20])
    config = TargetControlConfig(
        enable_fall_termination=False,
        enable_gap_termination=True,
        gap_termination_threshold=0.5,
        enable_stuck_termination=True,
        stuck_window_frames=4,
        stuck_movement_threshold=0.2,
        stuck_height_threshold=0.4,
        reset_grace_period=5,
    )
    control = TargetControl(config, inputs)
    state = make_robot_state([[0.0, 0.0, 0.1], [1.0, 0.0, 0.1]])
    control.reset_rows(torch.arange(2), state)
    control.last_support_root_height[:] = 1.0
    control.advance(state)
    published = control.publish(None)
    rules = TargetControl.terminations(config, inputs.robot)

    assert set(rules) == {"gap", "stuck"}
    # The gap check is a boolean kernel carrying the threshold itself (the old
    # ``z < last_support - threshold`` as written), so the rule has no cut.
    assert rules["gap"].threshold is None and not rules["gap"].truncation
    assert rules["gap"].component.static_params["gap_threshold"] == 0.5
    assert rules["stuck"].threshold is None
    assert torch.equal(published["target.last_support_root_height"], torch.ones(2))
    assert torch.equal(published["target.oldest_root_pos"], state.rigid_body_pos[:, 0])
    flags = _flags(
        rules,
        published,
        **{"current.anchor_pos": state.rigid_body_pos[:, 0]},
        progress_buf=inputs.clock.progress_buf,
    )
    # Env 0 is inside the grace period; env 1 dropped 0.9 m and did not move.
    assert torch.equal(flags["gap"], torch.tensor([False, True]))
    assert torch.equal(flags["stuck"], torch.tensor([False, True]))


def test_target_control_fall_check_combines_contact_height_and_grace():
    robot = make_robot(("torso", "foot"), non_termination_contact_bodies=["foot"])
    inputs = make_task_inputs(3, robot=robot, dt=0.1)
    inputs.clock.progress_buf[:] = torch.tensor([2, 10, 10])
    config = TargetControlConfig(
        enable_fall_termination=True,
        fall_termination_height=0.2,
        reset_grace_period=5,
    )
    control = TargetControl(config, inputs)
    state = make_robot_state(
        [[0.0, 0.0, 0.05], [1.0, 0.0, 0.05], [2.0, 0.0, 0.05]], num_bodies=2
    )
    control.reset_rows(torch.arange(3), state)
    control.advance(state)
    published = control.publish(None)
    rules = TargetControl.terminations(config, robot)
    layout = ContactLayout.build(["torso", "foot"], 0, 0)
    contacts = torch.tensor([[True, False], [True, False], [False, True]])
    forces = torch.zeros(3, 2, 3, 3)
    forces[:, :, 0, 2] = contacts.float()

    assert set(rules) == {"fall"}
    fall = rules["fall"].component
    assert fall.required_keys == {
        "contact_body_index": ["torso"],
        "contact_partner_index": ["terrain"],
    }
    assert torch.equal(fall.static_params["body_ids"], torch.tensor([0]))
    assert torch.equal(published["target.fall_ref_height"], torch.zeros(3))

    def run(forces):
        return _flags(
            rules,
            published,
            **{"current.rigid_body_pos": state.rigid_body_pos},
            contact_forces=forces,
            contact_body_index=layout.body_index,
            contact_partner_index=layout.partner_index,
            progress_buf=inputs.clock.progress_buf,
        )["fall"]

    assert torch.equal(run(forces), torch.tensor([False, True, False]))
    # Torso self-contact is not a terrain fall.
    forces[:, :, 1] = forces[:, :, 0]
    forces[:, :, 0] = 0
    assert not run(forces).any()


def test_target_control_fall_check_names_every_forbidden_body_for_the_build():
    """A forbidden body missing from the contact matrix is refused by the
    binding check through ``required_keys``, not by the control."""
    robot = make_robot(("torso", "foot"), non_termination_contact_bodies="all")
    rules = TargetControl.terminations(TargetControlConfig(), robot)
    assert rules["fall"].component.required_keys["contact_body_index"] == []
    robot = make_robot(
        ("torso", "arm", "foot"), non_termination_contact_bodies=["foot"]
    )
    rules = TargetControl.terminations(TargetControlConfig(), robot)
    assert rules["fall"].component.required_keys["contact_body_index"] == [
        "torso",
        "arm",
    ]
    assert TargetControl.provides(TargetControlConfig(), "failure_checks")
    assert not TargetControl.provides(
        TargetControlConfig(enable_fall_termination=False), "failure_checks"
    )


# ---------------------------------------------------------------------------
# Steering control
# ---------------------------------------------------------------------------


def test_steering_control_reset_advance_context_and_marker_state_are_deterministic():
    inputs = make_task_inputs(
        2, robot=make_robot(("root", "pelvis"), anchor_body_index=1), headless=False
    )
    control = SteeringControl(
        SteeringControlConfig(
            heading_change_steps_min=2,
            heading_change_steps_max=3,
            random_heading_probability=0.0,
            standard_heading_change=0.0,
            standard_speed_change=0.0,
            stop_probability=0.0,
            enable_rand_facing=False,
        ),
        inputs,
    )
    state = make_robot_state([[0.0, 0.0, 0.9], [2.0, 0.0, 0.9]], num_bodies=2)
    state.rigid_body_pos[:, 1] = torch.tensor([[1.0, 0.0, 0.9], [3.0, 0.0, 0.9]])

    control.reset_rows(torch.tensor([0, 1]), state)
    published = control.publish(None)

    assert torch.equal(control.heading_change_steps, torch.tensor([2, 2]))
    assert torch.allclose(
        published["steering.tar_dir"], torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    )
    assert torch.allclose(
        published["steering.tar_face_dir"], published["steering.tar_dir"]
    )
    assert torch.allclose(published["steering.tar_speed"], torch.ones(2))
    assert torch.allclose(
        published["steering.prev_root_pos"], state.rigid_body_pos[:, 1]
    )
    assert published_vocabulary_errors(published) == []
    assert SteeringControl.terminations(control.config, inputs.robot) == {}

    subset = control.publish(torch.tensor([1]))
    assert torch.equal(subset["steering.tar_dir"], control.tar_dir[[1]])
    assert torch.equal(subset["steering.prev_root_pos"], control.prev_root_pos[[1]])

    state.rigid_body_pos[:, 1, 0] += torch.tensor([0.5, 1.0])
    inputs.clock.progress_buf[:] = torch.tensor([1, 2])
    assert control.advance(state) == {}

    # Env 1's heading change was due: its commands were resampled, but the
    # double buffer still holds the previous step's position (3.0), not the
    # current one, so the velocity reward sees the true displacement.
    assert torch.allclose(
        control.prev_root_pos, torch.tensor([[1.0, 0.0, 0.9], [3.0, 0.0, 0.9]])
    )
    assert torch.allclose(
        control.curr_root_pos, torch.tensor([[1.5, 0.0, 0.9], [4.0, 0.0, 0.9]])
    )
    assert torch.equal(control.heading_change_steps, torch.tensor([2, 4]))
    marker_state = control.get_markers_state(state)
    assert set(marker_state) == {"movement_markers", "facing_markers"}
    assert torch.allclose(
        marker_state["movement_markers"].translation[:, 0],
        torch.tensor([[2.5, 0.0, 0.9], [5.0, 0.0, 0.9]]),
    )


def test_steering_control_empty_reset_keeps_state_and_declares_markers():
    inputs = make_task_inputs(1, headless=False)
    inputs.clock.progress_buf[:] = 123
    control = SteeringControl(SteeringControlConfig(), inputs)
    original_speed = control.tar_speed.clone()

    control.reset_rows(
        torch.tensor([], dtype=torch.long), make_robot_state([[0.0, 0.0, 0.9]])
    )
    markers = control.create_visualization_markers(headless=False)

    assert torch.equal(control.tar_speed, original_speed)
    assert set(markers) == {"movement_markers", "facing_markers"}
    assert markers["movement_markers"].type == "arrow"
    assert markers["facing_markers"].markers[0].size == "regular"


def test_steering_control_randomized_reset_stops_and_suppresses_headless_markers():
    torch.manual_seed(13)
    inputs = make_task_inputs(3, headless=True)
    inputs.clock.progress_buf[:] = torch.tensor([7, 8, 9])
    control = SteeringControl(
        SteeringControlConfig(
            tar_speed_min=0.25,
            tar_speed_max=1.75,
            heading_change_steps_min=3,
            heading_change_steps_max=6,
            random_heading_probability=1.0,
            standard_heading_change=0.0,
            standard_speed_change=0.0,
            stop_probability=1.0,
            enable_rand_facing=True,
        ),
        inputs,
    )
    state = make_robot_state([[0.0, 0.0, 0.9], [2.0, 0.0, 0.9], [4.0, 0.0, 0.9]])

    control.reset_rows(torch.tensor([0, 1, 2]), state)
    published = control.publish(None)

    assert torch.allclose(published["steering.tar_speed"], torch.zeros(3))
    assert torch.allclose(
        torch.linalg.norm(published["steering.tar_dir"], dim=-1), torch.ones(3)
    )
    assert torch.allclose(
        torch.linalg.norm(published["steering.tar_face_dir"], dim=-1), torch.ones(3)
    )
    # The change step counts from each env's own progress.
    assert torch.all(control.heading_change_steps >= inputs.clock.progress_buf + 3)
    assert torch.all(control.heading_change_steps < inputs.clock.progress_buf + 6)
    assert control.create_visualization_markers(headless=True) == {}
    assert control.get_markers_state(state) == {}


def test_steering_resample_leaves_the_anchor_double_buffer_alone():
    """A due heading change inside the episode samples commands only; the
    double buffer is episode-start state that only reset_rows seeds."""
    inputs = make_task_inputs(2)
    control = SteeringControl(
        SteeringControlConfig(heading_change_steps_min=1, heading_change_steps_max=2),
        inputs,
    )
    control.reset_rows(
        torch.arange(2), make_robot_state([[0.0, 0.0, 0.9], [1.0, 0.0, 0.9]])
    )
    control.advance(make_robot_state([[0.1, 0.0, 0.9], [1.1, 0.0, 0.9]]))
    inputs.clock.progress_buf[:] = 5  # every change is due
    before = control.curr_root_pos.clone()
    control.advance(make_robot_state([[0.3, 0.0, 0.9], [1.3, 0.0, 0.9]]))
    assert torch.all(control.heading_change_steps >= 5)
    torch.testing.assert_close(control.prev_root_pos, before)
    torch.testing.assert_close(control.curr_root_pos[:, 0], torch.tensor([0.3, 1.3]))


def test_steering_resample_walks_from_the_previous_heading_and_speed():
    """The row reset is deliberately not the registry's clear: the incremental
    branch reads the previous episode's heading and speed."""
    inputs = make_task_inputs(2)
    control = SteeringControl(
        SteeringControlConfig(
            random_heading_probability=0.0,
            standard_heading_change=0.0,
            standard_speed_change=0.0,
            stop_probability=0.0,
            enable_rand_facing=False,
        ),
        inputs,
    )
    control.tar_dir_theta[:] = torch.tensor([0.5, -1.0])
    control.tar_speed[:] = torch.tensor([1.5, 0.75])
    control.reset_rows(
        torch.arange(2), make_robot_state([[0.0, 0.0, 0.9], [1.0, 0.0, 0.9]])
    )
    assert torch.allclose(control.tar_dir_theta, torch.tensor([0.5, -1.0]))
    assert torch.allclose(control.tar_speed, torch.tensor([1.5, 0.75]))


# ---------------------------------------------------------------------------
# Path follower
# ---------------------------------------------------------------------------


def _path_robot():
    return make_robot(
        ("root", "spine", "head"), anchor_body_index=1, head_body_name="head"
    )


def test_path_follower_reset_context_and_marker_samples(monkeypatch):
    import protomotions.envs.control.path_follower_control as path_module

    class _FakePathGenerator:
        instances = []

        def __init__(
            self, config, state, generator, num_envs, episode_dur, height_conditioned
        ):
            self.reset_calls = []
            self.episode_dur = episode_dur
            _FakePathGenerator.instances.append(self)

        def reset(self, env_ids, head_position):
            self.reset_calls.append((env_ids.clone(), head_position.clone()))

        def calc_pos(self, env_ids, times):
            env_ids = env_ids.float()
            return torch.stack((env_ids, times, env_ids + times), dim=-1)

    monkeypatch.setattr(path_module, "PathGenerator", _FakePathGenerator)
    inputs = make_task_inputs(
        2,
        robot=_path_robot(),
        dt=0.5,
        max_episode_length=20,
        ground_height=0.2,
        headless=False,
    )
    inputs.clock.progress_buf[:] = torch.tensor([2, 4])
    state = make_robot_state([[1.0, 2.0, 1.5], [3.0, 4.0, 1.7]], num_bodies=3)
    state.rigid_body_pos[:, 1] = torch.tensor([[5.0, 6.0, 2.5], [7.0, 8.0, 2.7]])
    state.rigid_body_pos[:, 2] = torch.tensor([[1.5, 2.5, 2.0], [3.5, 4.5, 2.4]])
    config = PathFollowerControlConfig(num_traj_samples=3, traj_sample_timestep=0.25)
    config.path_generator.height_conditioned = False
    control = PathFollowerControl(config, inputs)
    assert control.path_generator.episode_dur == 10.0

    control.reset_rows(torch.tensor([0, 1]), state)
    published = control.publish(None)
    markers = control.get_markers_state(state)

    reset_ids, reset_head_pos = control.path_generator.reset_calls[0]
    assert torch.equal(reset_ids, torch.tensor([0, 1]))
    assert torch.allclose(reset_head_pos[:, :2], torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
    assert torch.allclose(reset_head_pos[:, 2], torch.tensor([2.3, 2.5]))
    assert control.head_body_id == 2
    assert published["path.height_conditioned"] is False
    assert torch.allclose(
        published["path.tar_pos"], torch.tensor([[0.0, 1.0, 1.0], [1.0, 2.0, 3.0]])
    )
    assert torch.allclose(published["path.head_pos"][:, 2], torch.tensor([0.5, 0.7]))
    assert published["path.traj_samples"].shape == (2, 3, 3)
    assert published_vocabulary_errors(published) == []
    assert torch.allclose(
        markers["path_markers"].translation[..., 2], torch.full((2, 3), 1.0)
    )
    assert torch.equal(markers["path_markers"].orientation, torch.zeros(2, 3, 4))

    subset = control.publish(torch.tensor([1]))
    assert subset["path.tar_pos"].shape == (1, 3)
    assert subset["path.traj_samples"].shape == (1, 3, 3)


def test_path_follower_checks_fire_on_distance_and_height_only_after_min_progress(
    monkeypatch,
):
    import protomotions.envs.control.path_follower_control as path_module

    class _StaticPathGenerator:
        def __init__(
            self, config, state, generator, num_envs, episode_dur, height_conditioned
        ):
            pass

        def reset(self, env_ids, head_position):
            pass

        def calc_pos(self, env_ids, times):
            return torch.zeros(len(env_ids), 3)

    monkeypatch.setattr(path_module, "PathGenerator", _StaticPathGenerator)
    robot = make_robot(("root", "head"), head_body_name="head")
    inputs = make_task_inputs(2, robot=robot, dt=0.1, max_episode_length=20)
    inputs.clock.progress_buf[:] = torch.tensor([10, 11])
    state = make_robot_state([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], num_bodies=2)
    state.rigid_body_pos[:, 0] = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    state.rigid_body_pos[:, 1] = torch.tensor([[0.0, 0.0, 2.0], [2.0, 0.0, 2.0]])
    config = PathFollowerControlConfig(
        enable_path_termination=True, fail_dist=1.0, fail_height_dist=0.5
    )
    config.path_generator.height_conditioned = True
    control = PathFollowerControl(config, inputs)
    control.advance(state)
    rules = PathFollowerControl.terminations(config, robot)

    assert set(rules) == {"path_distance", "path_height"}
    # Both are failures (not truncations): a failed check resets the env too.
    assert not rules["path_distance"].truncation and not rules["path_height"].truncation
    flags = _flags(rules, control.publish(None), progress_buf=inputs.clock.progress_buf)
    assert torch.equal(flags["path_distance"], torch.tensor([False, True]))
    assert torch.equal(flags["path_height"], torch.tensor([False, True]))
    assert control.create_visualization_markers(headless=True) == {}


def test_path_follower_disabled_termination_owns_no_check():
    config = PathFollowerControlConfig(enable_path_termination=False)
    config.path_generator.height_conditioned = True
    assert PathFollowerControl.terminations(config, make_robot()) == {}
    flat = PathFollowerControlConfig(enable_path_termination=True)
    assert set(PathFollowerControl.terminations(flat, make_robot())) == {
        "path_distance"
    }


def test_path_follower_height_conditioned_markers_and_default_queries(monkeypatch):
    import protomotions.envs.control.path_follower_control as path_module

    class _RecordingPathGenerator:
        def __init__(
            self, config, state, generator, num_envs, episode_dur, height_conditioned
        ):
            del config, state, generator, num_envs, episode_dur
            self.height_conditioned = height_conditioned
            self.reset_calls = []
            self.calc_calls = []

        def reset(self, env_ids, head_position):
            self.reset_calls.append((env_ids.clone(), head_position.clone()))

        def calc_pos(self, env_ids, times):
            self.calc_calls.append((env_ids.clone(), times.clone()))
            return torch.stack((env_ids.float() + 0.5, times, times + 1.0), dim=-1)

    monkeypatch.setattr(path_module, "PathGenerator", _RecordingPathGenerator)
    robot = make_robot(("root", "head"), head_body_name="head")
    inputs = make_task_inputs(
        2,
        robot=robot,
        dt=0.5,
        max_episode_length=20,
        ground_height=0.25,
        headless=False,
    )
    inputs.clock.progress_buf[:] = torch.tensor([2, 4])
    state = make_robot_state([[0.0, 0.0, 0.5], [1.0, 1.0, 0.25]], num_bodies=2)
    state.rigid_body_pos[:, 1] = torch.tensor([[0.0, 0.0, 1.5], [1.0, 1.0, 2.25]])
    config = PathFollowerControlConfig(num_traj_samples=2, traj_sample_timestep=0.5)
    config.path_generator.height_conditioned = True
    control = PathFollowerControl(config, inputs)

    control.reset_rows(torch.tensor([], dtype=torch.long), state)
    assert control.path_generator.reset_calls == []

    control.reset_rows(torch.tensor([0, 1]), state)
    reset_ids, reset_head_pos = control.path_generator.reset_calls[0]
    assert torch.equal(reset_ids, torch.tensor([0, 1]))
    torch.testing.assert_close(reset_head_pos[:, 2], torch.tensor([0.25, 0.0]))
    assert control.advance(state) == {}
    torch.testing.assert_close(
        control.calc_target_pos(), torch.tensor([[0.5, 1.0, 2.0], [1.5, 2.0, 3.0]])
    )
    torch.testing.assert_close(control.head_pos[:, 2], torch.tensor([1.0, 2.0]))

    marker_cfg = control.create_visualization_markers(headless=False)
    marker_state = control.get_markers_state(state)

    assert len(marker_cfg["path_markers"].markers) == 2
    torch.testing.assert_close(
        marker_state["path_markers"].translation[..., 2],
        torch.tensor([[2.25, 2.75], [3.25, 3.75]]),
    )
    assert control.create_visualization_markers(headless=True) == {}


# ---------------------------------------------------------------------------
# GPC reference target
# ---------------------------------------------------------------------------


def test_gpc_reference_target_context_selects_reset_subset():
    control = GPCSFTReferenceTargetControl(
        GPCSFTReferenceTargetControlConfig(tar_proximity_threshold=0.25),
        make_task_inputs(3),
    )
    control.tar_pos[:] = torch.arange(9, dtype=torch.float).reshape(3, 3)

    published = control.publish(torch.tensor([2, 0]))

    assert torch.equal(published["target.tar_pos"], control.tar_pos[[2, 0]])
    assert published["target.tar_proximity_threshold"] == 0.25
    assert published_vocabulary_errors(published) == []
    assert GPCSFTReferenceTargetControl.terminations(control.config, make_robot()) == {}


# ---------------------------------------------------------------------------
# The two animation players
# ---------------------------------------------------------------------------


def test_external_kinematic_control_queues_one_pending_pose_and_restarts_episodes():
    inputs = make_task_inputs(2)
    inputs.clock.progress_buf[:] = torch.tensor([9, 8])
    control = ExternalKinematicControl(ExternalKinematicControlConfig(), inputs)
    pending_pose = make_reset_state(num_envs=2)
    state = make_robot_state([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    control.set_next_pose(pending_pose)
    assert control.advance(state) == {}
    assert inputs.teleport.requests == []

    control.reset_rows(torch.tensor([0, 1]), state)
    control.set_next_pose(pending_pose)
    assert control.advance(state) == {}
    published = control.publish(None)

    env_ids, reset_state, object_state = inputs.teleport.requests[0]
    assert reset_state is pending_pose
    assert object_state is None
    assert torch.equal(env_ids, torch.tensor([0, 1]))
    assert torch.equal(inputs.clock.progress_buf, torch.tensor([0, 0]))
    assert control._next_pose is None
    assert published == {}
    assert ExternalKinematicControl.terminations(control.config, inputs.robot) == {}


def test_kinematic_replay_advances_motion_resamples_done_and_offsets_states():
    class _MotionManager:
        def __init__(self):
            self.motion_ids = torch.tensor([3, 4])
            self.motion_times = torch.tensor([0.5, 0.8])
            self.sample_calls = []

        # The replay restarts clips through the query's assign (the real
        # in-place writer over this fake's tensors).
        assign = MotionManager.assign

        def sample_motions(self, env_ids):
            self.sample_calls.append(env_ids.clone())

        def get_done_tracks(self, env_ids=None):
            return torch.tensor([False, True])

    class _MotionLib:
        def get_motion_state(self, motion_ids, motion_times):
            num_envs = len(motion_ids)
            return RobotState(
                state_conversion=StateConversion.COMMON,
                rigid_body_pos=torch.stack(
                    (
                        torch.stack(
                            (motion_ids.float(), motion_times, torch.zeros(num_envs)),
                            dim=-1,
                        ),
                        torch.ones(num_envs, 3),
                    ),
                    dim=1,
                ),
                rigid_body_rot=identity_quat(num_envs, 2),
                rigid_body_vel=torch.ones(num_envs, 2, 3) * 7.0,
                rigid_body_ang_vel=torch.ones(num_envs, 2, 3) * 8.0,
                dof_pos=motion_times.unsqueeze(-1),
                dof_vel=torch.ones(num_envs, 1) * 9.0,
            )

    class _SceneLib:
        def __init__(self):
            self.pose_queries = []

        def get_scene_pose(self, env_ids, motion_times, offset):
            self.pose_queries.append((env_ids.clone(), motion_times.clone(), offset))
            return make_object_state(
                torch.zeros(len(env_ids), 1, 3), root_vel=torch.ones(len(env_ids), 1, 3)
            )

        def num_scenes(self):
            return 1

    motion_manager = _MotionManager()
    scene_lib = _SceneLib()
    motion = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        motion_manager=motion_manager,
        motion_lib=_MotionLib(),
    )
    reference = fake_reference(
        motion,
        placement_offset=lambda root, env_ids: torch.ones(root.shape[0], 1, 3),
        scene_lib=scene_lib,
        ref_object_respawn_offset=0.3,
    )
    # The replay's clock tick is the env step (decimation / fps = 0.1 here).
    inputs = make_task_inputs(2, reference=reference, dt=0.1)
    inputs.clock.progress_buf[:] = torch.tensor([5, 6])
    inputs.ui.handle_key_event("R", pressed=True)
    control = KinematicReplayControl(KinematicReplayControlConfig(), inputs)
    # Replay drives the simulator through the teleport channel and publishes nothing.
    assert control.publish(None) == {}

    assert control.advance(make_robot_state([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])) == {}
    env_ids, reset_state, object_state = inputs.teleport.requests[0]

    assert torch.equal(motion_manager.sample_calls[0], torch.tensor([0, 1]))
    assert torch.equal(motion_manager.sample_calls[1], torch.tensor([1]))
    assert torch.allclose(motion_manager.motion_times, torch.tensor([0.1, 0.1]))
    assert torch.equal(env_ids, torch.tensor([0, 1]))
    assert torch.allclose(
        reset_state.root_pos, torch.tensor([[4.0, 1.1, 1.0], [5.0, 1.1, 1.0]])
    )
    assert torch.equal(reset_state.dof_vel, torch.zeros(2, 1))
    assert torch.equal(reset_state.root_vel, torch.zeros(2, 3))
    assert torch.equal(object_state.root_vel, torch.zeros(2, 1, 3))
    assert torch.allclose(object_state.root_pos, torch.ones(2, 1, 3))
    assert torch.equal(inputs.clock.progress_buf, torch.tensor([0, 0]))
    # The viewer's press was consumed by the replay, not left for the env.
    assert not inputs.viewer_reset.pressed()


def test_kinematic_replay_without_scenes_writes_no_object_state():
    """Without scenes the replay passes no object state to the teleport (the
    query answers None), and the robot pose is still placed."""

    class _MotionManager:
        motion_ids = torch.tensor([2])
        motion_times = torch.tensor([0.4])

        def __init__(self):
            self.sample_calls = []

        def sample_motions(self, env_ids):
            self.sample_calls.append(env_ids.clone())

        def get_done_tracks(self, env_ids=None):
            return torch.tensor([False])

    class _MotionLib:
        def get_motion_state(self, motion_ids, motion_times):
            return RobotState(
                state_conversion=StateConversion.COMMON,
                rigid_body_pos=torch.tensor([[[2.0, motion_times[0].item(), 0.0]]]),
                rigid_body_rot=identity_quat(1, 1),
                rigid_body_vel=torch.ones(1, 1, 3),
                rigid_body_ang_vel=torch.ones(1, 1, 3),
                dof_pos=torch.ones(1, 1),
                dof_vel=torch.ones(1, 1),
            )

    class _SceneLib:
        def get_scene_pose(self, env_ids, motion_times, offset):
            del env_ids, motion_times, offset
            return make_object_state(torch.ones(1, 1, 3) * 5.0)

        def num_scenes(self):
            return 0

    motion_manager = _MotionManager()
    motion = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        motion_manager=motion_manager,
        motion_lib=_MotionLib(),
    )
    reference = fake_reference(
        motion,
        placement_offset=lambda root, env_ids: torch.ones(root.shape[0], 1, 3),
        scene_lib=_SceneLib(),
    )
    inputs = make_task_inputs(1, reference=reference, dt=0.1)
    inputs.clock.progress_buf[:] = 7
    control = KinematicReplayControl(KinematicReplayControlConfig(), inputs)

    control.advance(make_robot_state([[0.0, 0.0, 0.0]]))
    env_ids, reset_state, object_state = inputs.teleport.requests[0]

    assert motion_manager.sample_calls == []
    assert torch.equal(env_ids, torch.tensor([0]))
    assert torch.allclose(reset_state.root_pos, torch.tensor([[3.0, 1.5, 1.0]]))
    assert object_state is None
    assert torch.equal(inputs.clock.progress_buf, torch.tensor([0]))


# ---------------------------------------------------------------------------
# Masked mimic
# ---------------------------------------------------------------------------


def _masked_mimic_inputs(
    motion_manager,
    motion_lib,
    *,
    body_names=("root", "hand", "foot"),
    trackable=("hand", "foot"),
    small=frozenset({"foot"}),
    hinge_axes_map=None,
    placement_offset=None,
    headless=True,
):
    num_envs = len(motion_manager.motion_ids)
    motion = SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        motion_manager=motion_manager,
        motion_lib=motion_lib,
    )
    robot = make_robot(
        body_names,
        trackable_bodies_subset=trackable,
        mimic_small_marker_bodies=small,
        hinge_axes_map=(
            {1: torch.tensor([[0.0, 0.0, 1.0]])}
            if hinge_axes_map is None
            else hinge_axes_map
        ),
    )
    return make_task_inputs(
        num_envs,
        robot=robot,
        reference=fake_reference(motion, placement_offset=placement_offset),
        dt=0.1,
        headless=headless,
    )


def _zero_pose_lib(lengths):
    class _MotionLib:
        motion_lengths = torch.as_tensor(lengths)

        def get_motion_length(self, motion_ids):
            return self.motion_lengths[motion_ids]

        def get_motion_state(self, motion_ids, motion_times):
            num = len(motion_ids)
            return RobotState(
                state_conversion=StateConversion.COMMON,
                rigid_body_pos=torch.zeros(num, 3, 3),
                rigid_body_rot=identity_quat(num, 3),
                rigid_body_vel=torch.zeros(num, 3, 3),
                rigid_body_ang_vel=torch.zeros(num, 3, 3),
                dof_pos=torch.zeros(num, 1),
                dof_vel=torch.zeros(num, 1),
            )

    return _MotionLib()


def test_masked_mimic_fixed_masks_context_and_colored_marker_state():
    class _MotionLib:
        motion_lengths = torch.tensor([2.0, 2.0])

        def get_motion_length(self, motion_ids):
            return self.motion_lengths[motion_ids]

        def get_motion_state(self, motion_ids, motion_times):
            num = len(motion_ids)
            pos = torch.zeros(num, 3, 3)
            pos[:, :, 0] = motion_ids.float().unsqueeze(-1)
            pos[:, :, 1] = motion_times.unsqueeze(-1)
            pos[:, :, 2] = torch.arange(3).float()
            return RobotState(
                state_conversion=StateConversion.COMMON,
                rigid_body_pos=pos,
                rigid_body_rot=identity_quat(num, 3),
                rigid_body_vel=torch.zeros(num, 3, 3),
                rigid_body_ang_vel=torch.zeros(num, 3, 3),
                dof_pos=motion_times.unsqueeze(-1),
                dof_vel=torch.zeros(num, 1),
            )

    inputs = _masked_mimic_inputs(
        SimpleNamespace(
            motion_ids=torch.tensor([0, 1]),
            motion_times=torch.tensor([0.2, 0.2]),
            get_done_tracks=lambda env_ids=None: torch.tensor([False, False]),
        ),
        _MotionLib(),
        hinge_axes_map={1: torch.tensor([[0.0, 0.0, 1.0]])},
        placement_offset=lambda ref, env_ids: torch.ones(ref.shape[0], 1, 3),
        headless=False,
    )
    config = MaskedMimicControlConfig(
        future_steps=1,
        num_masked_future_steps=2,
        fixed_conditioning=[FixedBodyCondition("hand", 1)],
        visible_target_pose_prob=1.0,
    )
    control = MaskedMimicControl(config, inputs)
    state = make_robot_state(torch.zeros(2, 3, 3))

    control.global_stream.times[:] = torch.tensor([[1.4, 1.8], [0.6, 1.4]])
    control.global_stream.frame_masks[:] = torch.tensor([[True, False], [True, False]])
    control.global_stream.body_masks[:] = torch.tensor(
        [
            [True, False, False, False, False, False, False, False],
            [True, False, False, False, False, False, False, False],
        ]
    )
    control.planned[:] = True

    marker_cfg = control.create_visualization_markers(headless=False)
    markers = control.get_markers_state(state)
    published = control.publish(None)

    assert [marker.size for marker in marker_cfg["body_markers_blue"].markers] == [
        "regular",
        "small",
    ]
    assert set(markers) == {
        "body_markers_blue",
        "body_markers_yellow",
        "body_markers_red",
    }
    assert torch.allclose(
        markers["body_markers_blue"].translation[0, 0], torch.tensor([1.0, 2.4, 2.0])
    )
    assert torch.all(markers["body_markers_blue"].translation[1] > 50.0)
    assert torch.allclose(
        markers["body_markers_yellow"].translation[1, 0], torch.tensor([2.0, 1.6, 2.0])
    )
    assert torch.all(markers["body_markers_red"].translation > 50.0)
    assert published["masked_mimic.ref_pos"].shape == (2, 2, 3, 3)
    assert torch.allclose(
        published["masked_mimic.time_offsets"], torch.tensor([[1.2, 1.6], [0.4, 1.2]])
    )
    assert torch.equal(
        published["masked_mimic.target_poses_masks"], control.global_stream.frame_masks
    )
    assert torch.equal(
        published["masked_mimic.target_bodies_masks"], control.global_stream.body_masks
    )
    # A disabled modality is not published (its requires= switch is off), so
    # a binding to it is refused by name before the kernel runs.
    assert "masked_mimic.local_ref_pos" not in published
    assert not any(key.startswith("masked_mimic.style_") for key in published)
    # The clip-done flag is published for the parent's check.
    assert torch.equal(published["mimic.clip_done"], torch.tensor([False, False]))
    # Every mimic.* and masked_mimic.* name is declared, every always-present
    # one is there.
    assert published_vocabulary_errors(published) == []

    subset = control.publish(torch.tensor([1]))
    assert subset["mimic.ref_state.rigid_body_pos"].shape[0] == 1
    assert subset["masked_mimic.ref_pos"].shape == (1, 2, 3, 3)
    assert torch.equal(
        subset["masked_mimic.target_times"], control.global_stream.times[[1]]
    )

    unplanned = MaskedMimicControl(config, inputs)
    assert unplanned.get_markers_state(state) == {}


def test_masked_mimic_publishes_every_modality_its_switches_turn_on():
    """The ``requires=`` tags on the ``masked_mimic.*`` names promise that a
    switch which is on means the name is published. With all four switches
    on, every name the rule says this configuration publishes is in the
    dict, nothing it says is off is there, and no tag is violated."""

    class _MotionLib:
        motion_lengths = torch.tensor([2.0, 2.0])

        def get_motion_length(self, motion_ids):
            return self.motion_lengths[motion_ids]

        def get_motion_state(self, motion_ids, motion_times):
            num = len(motion_ids)
            pos = torch.zeros(num, 3, 3)
            pos[:, :, 1] = motion_times.unsqueeze(-1)
            return RobotState(
                state_conversion=StateConversion.COMMON,
                rigid_body_pos=pos,
                rigid_body_rot=identity_quat(num, 3),
                rigid_body_vel=torch.zeros(num, 3, 3),
                rigid_body_ang_vel=torch.zeros(num, 3, 3),
                dof_pos=motion_times.unsqueeze(-1),
                dof_vel=torch.zeros(num, 1),
            )

    inputs = _masked_mimic_inputs(
        SimpleNamespace(
            motion_ids=torch.tensor([0, 1]),
            motion_times=torch.tensor([0.2, 0.4]),
            get_done_tracks=lambda env_ids=None: torch.tensor([False, False]),
        ),
        _MotionLib(),
        # The anchor must be trackable for anchor, local and style goals.
        trackable=("root", "hand", "foot"),
        small=frozenset(),
        hinge_axes_map={1: torch.tensor([[0.0, 0.0, 1.0]])},
    )
    config = MaskedMimicControlConfig(
        future_steps=1,
        num_masked_future_steps=2,
        local_target_condition_probability=1.0,
        anchor_pose_submasking=True,
        style=MaskedMimicStyleConfig(),
        velocity_command=MaskedMimicVelocityCommandConfig(),
    )
    control = MaskedMimicControl(config, inputs)
    torch.manual_seed(5)
    control.reset_rows(torch.arange(2), make_robot_state(torch.zeros(2, 3, 3)))
    published = control.publish(None)

    facts = Facts(
        history_steps=0,
        contacts=False,
        sensors=None,
        terrain_heightfield=False,
        control={"masked_mimic": (MaskedMimicControl, config)},
    )
    assert published_vocabulary_errors(published) == []
    assert requirement_errors(published, facts) == []
    for namespace in MaskedMimicControl.namespaces:
        for path in declared_paths(namespace):
            verdict = status(path, facts)[0]
            if verdict is True:
                assert path in published, path
            elif verdict is False:
                assert path not in published, path
    assert "masked_mimic.style_pose_samples" in published
    assert "masked_mimic.local_ref_pos" in published
    assert "masked_mimic.anchor_rotation_mask" in published
    assert "masked_mimic.velocity_command_mask" in published
    # The one mimic switch left off is the one mimic name left out.
    assert "mimic.smoothed_ref_anchor_height" not in published


def test_masked_mimic_reset_plans_future_times_and_can_hide_all_targets():
    motion_manager = SimpleNamespace(
        motion_ids=torch.tensor([0, 1]),
        motion_times=torch.tensor([0.2, 1.4]),
        get_done_tracks=lambda env_ids=None: torch.tensor([False, False]),
    )
    inputs = _masked_mimic_inputs(
        motion_manager, _zero_pose_lib([1.0, 1.5]), small=frozenset()
    )
    control = MaskedMimicControl(
        MaskedMimicControlConfig(
            future_steps=1,
            num_masked_future_steps=3,
            visible_target_pose_prob=0.0,
            target_profile_probabilities=(0, 1, 0),
        ),
        inputs,
    )
    assert not control.planned.any()

    torch.manual_seed(23)
    control.reset_rows(torch.tensor([0, 1]), make_robot_state(torch.zeros(2, 3, 3)))

    assert control.planned.all()
    assert control.global_stream.times.shape == (2, 3)
    assert torch.all(
        control.global_stream.times >= motion_manager.motion_times.unsqueeze(-1)
    )
    assert torch.all(control.global_stream.times <= torch.tensor([[1.0], [1.5]]))
    assert torch.equal(
        control.global_stream.frame_masks, torch.zeros(2, 3, dtype=torch.bool)
    )
    assert torch.equal(
        control.global_stream.body_masks, torch.zeros(2, 3 * 2 * 2, dtype=torch.bool)
    )


def test_masked_mimic_body_mask_sampling_honors_dense_and_sparse_profiles():
    inputs = _masked_mimic_inputs(
        SimpleNamespace(
            motion_ids=torch.zeros(4, dtype=torch.long),
            motion_times=torch.zeros(4),
            get_done_tracks=lambda env_ids=None: torch.zeros(4, dtype=torch.bool),
        ),
        SimpleNamespace(motion_lengths=torch.ones(1)),
        body_names=("root", "a", "b", "c", "d", "e"),
        trackable=("a", "b", "c", "d", "e"),
        small=frozenset(),
    )
    max_control = MaskedMimicControl(
        MaskedMimicControlConfig(
            num_masked_future_steps=1, target_profile_probabilities=(1, 0, 0)
        ),
        inputs,
    )

    torch.manual_seed(29)
    max_masks = max_control._sample_target_plan(4)[1].view(4, 5, 2)
    assert torch.equal(max_masks.any(dim=-1), torch.ones(4, 5, dtype=torch.bool))

    small_control = MaskedMimicControl(
        MaskedMimicControlConfig(
            num_masked_future_steps=1, target_profile_probabilities=(0, 0, 1)
        ),
        inputs,
    )

    torch.manual_seed(31)
    small_masks = small_control._sample_target_plan(4)[1].view(4, 5, 2)
    active_counts = small_masks.any(dim=-1).sum(dim=-1)
    assert torch.all(active_counts == 1)


def test_masked_mimic_keeps_episode_masks_and_inherits_clip_termination():
    motion_manager = SimpleNamespace(
        motion_ids=torch.tensor([0, 1]),
        motion_times=torch.tensor([0.5, 0.5]),
        get_done_tracks=lambda env_ids=None: torch.tensor([False, True]),
    )
    lib = _zero_pose_lib([2.0, 3.0])
    inputs = _masked_mimic_inputs(motion_manager, lib, small=frozenset())
    config = MaskedMimicControlConfig(
        bootstrap_on_episode_end=False,
        future_steps=1,
        num_masked_future_steps=3,
        target_profile_probabilities=(1, 0, 0),
        visible_target_pose_prob=1.0,
        fixed_conditioning=[FixedBodyCondition("hand", 0)],
    )
    control = MaskedMimicControl(config, inputs)
    state = make_robot_state(torch.zeros(2, 3, 3))
    torch.manual_seed(17)
    control.reset_rows(torch.arange(2), state)
    masks = control.global_stream.body_masks.clone()
    times = control.global_stream.times.clone()
    motion_manager.motion_times += 0.1
    assert control.advance(state) == {}
    published = control.publish(None)
    rules = MaskedMimicControl.terminations(config, inputs.robot)

    assert torch.equal(control.global_stream.body_masks, masks)
    assert control.global_stream.frame_masks.all()
    assert torch.all(control.global_stream.times >= times)
    assert torch.all(control.global_stream.times <= lib.motion_lengths[:, None])
    # The parent's one check: with bootstrapping off a finished clip is a failure.
    assert set(rules) == {"clip_done"}
    assert rules["clip_done"].truncation is False
    flags = _flags(rules, published)
    assert torch.equal(flags["clip_done"], torch.tensor([False, True]))
    assert MaskedMimicControl.terminations(
        MaskedMimicControlConfig(bootstrap_on_episode_end=True), inputs.robot
    )["clip_done"].truncation
    assert control.create_visualization_markers(headless=True) == {}
    assert control.get_markers_state(state) == {}


def test_masked_mimic_streams_rebase_once_any_row_is_planned():
    """Before the first reset no stream is rebased; from the first reset on
    every due env's stream is (the reference behaviour's one flag)."""
    motion_manager = SimpleNamespace(
        motion_ids=torch.tensor([0, 1]),
        motion_times=torch.tensor([0.5, 0.5]),
        get_done_tracks=lambda env_ids=None: torch.tensor([False, False]),
    )
    inputs = _masked_mimic_inputs(
        motion_manager, _zero_pose_lib([2.0, 3.0]), small=frozenset()
    )
    control = MaskedMimicControl(
        MaskedMimicControlConfig(future_steps=1, num_masked_future_steps=2), inputs
    )
    state = make_robot_state(torch.zeros(2, 3, 3))
    control.advance(state)
    assert torch.all(control.global_stream.times == 0)
    torch.manual_seed(3)
    control.reset_rows(torch.tensor([1]), state)
    control.advance(state)
    assert torch.equal(control.planned, torch.tensor([False, True]))
    assert torch.all(control.global_stream.times >= 0.5)
