# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The MaskedMimic goal canvas: one derived layout, checked against the bound
paths at build (handover card C2)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.base import (
    ControlComponent,
    ControlComponentConfig,
    bound_paths,
)
from protomotions.envs.control.masked_mimic_control import (
    MaskedMimicControl,
    MaskedMimicControlConfig,
    MaskedMimicStyleConfig,
    MaskedMimicVelocityCommandConfig,
)
from protomotions.envs.control.masked_mimic_prompt import (
    ANCHOR_IN_TRACKABLE,
    GETUP_NEEDS_LOCAL_STREAM,
    SUBMASKING_UNBOUND,
    GoalCanvas,
)
from protomotions.envs.control.masked_mimic_tasks.controller import (
    GetupRecoveryConfig,
    MaskedMimicTaskControl,
    configure_masked_mimic_task,
)
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.rewards.meta import RewardMeta


def _robot(
    body_names=("pelvis", "torso", "hand"), anchor=1, trackable=("torso", "hand")
):
    return SimpleNamespace(
        kinematic_info=SimpleNamespace(body_names=list(body_names)),
        anchor_body_index=anchor,
        trackable_bodies_subset=list(trackable),
    )


def _binding(path, meta=None):
    return MdpComponent(lambda x: x, {"x": path}, meta=meta)


FULL = MaskedMimicControlConfig(
    num_masked_future_steps=4,
    future_steps=[1, 3, 5],
    local_target_condition_probability=0.5,
    anchor_pose_submasking=True,
    style=MaskedMimicStyleConfig(
        num_frames=3, visible_frame_count_probabilities=(0.25,) * 4
    ),
    velocity_command=MaskedMimicVelocityCommandConfig(),
)


def test_derive_reads_the_smpl_transformer_layout():
    """On the SMPL robot with the transformer recipe's switches (global targets
    only) the canvas is the body subset, the anchor by name, no optional
    stream, and the full stream's step indices."""
    from protomotions.robot_configs.factory import robot_config

    smpl = robot_config("smpl")
    canvas = GoalCanvas.derive(
        MaskedMimicControlConfig(num_masked_future_steps=10), smpl
    )
    assert canvas.anchor_body == smpl.kinematic_info.body_names[smpl.anchor_body_index]
    assert canvas.conditionable_bodies == tuple(smpl.trackable_bodies_subset)
    assert canvas.num_masked_future_steps == 10
    assert canvas.future_steps == (1,)
    assert not canvas.anchor_pose_submasking
    assert not canvas.has_local_stream
    assert canvas.style_frames is None
    assert not canvas.has_velocity_command


def test_derive_turns_every_stream_on_and_normalises_the_future_steps():
    """Local, style and velocity switches and a step list all reach the canvas."""
    canvas = GoalCanvas.derive(FULL, _robot())
    assert canvas == GoalCanvas(
        future_steps=(1, 3, 5),
        num_masked_future_steps=4,
        conditionable_bodies=("torso", "hand"),
        anchor_body="torso",
        anchor_pose_submasking=True,
        has_local_stream=True,
        style_frames=3,
        has_velocity_command=True,
    )
    assert GoalCanvas.derive(
        MaskedMimicControlConfig(future_steps=3), _robot()
    ).future_steps == (1, 2, 3)


def test_derive_normalises_the_int_a_cli_override_leaves_behind():
    """The override funnel writes after ``__post_init__``, so the saved config
    may hold the int form; the canvas still sees the tuple."""
    config = MaskedMimicControlConfig()
    config.future_steps = 3
    assert GoalCanvas.derive(config, _robot()).future_steps == (1, 2, 3)


def test_derive_refuses_an_anchor_the_goals_need_but_cannot_condition():
    """Anchor, local or style goals need the anchor in trackable_bodies_subset."""
    for config in (
        MaskedMimicControlConfig(anchor_pose_submasking=True),
        MaskedMimicControlConfig(local_target_condition_probability=1.0),
        MaskedMimicControlConfig(style=MaskedMimicStyleConfig()),
    ):
        with pytest.raises(ValueError, match=re.escape(ANCHOR_IN_TRACKABLE)):
            GoalCanvas.derive(config, _robot(trackable=("hand",)))
    # Global targets alone do not need the anchor.
    GoalCanvas.derive(MaskedMimicControlConfig(), _robot(trackable=("hand",)))


def test_validate_refuses_what_the_layout_cannot_serve():
    """The three refusals: a get-up preset without a local stream, a binding to
    a stream the canvas lacks, and anchor sub-masking nothing reads."""
    plain = GoalCanvas.derive(MaskedMimicControlConfig(), _robot())
    with pytest.raises(ValueError, match=re.escape(GETUP_NEEDS_LOCAL_STREAM)):
        plain.validate(frozenset(), getup=True)
    for path, missing in (
        ("masked_mimic.local_ref_pos", "local stream"),
        ("masked_mimic.style_frame_masks", "style stream"),
        ("masked_mimic.average_velocity", "velocity command"),
        (
            "masked_mimic.anchor_rotation_mask",
            "anchor sub-masks (anchor_pose_submasking is off)",
        ),
    ):
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"observation binds '{path}' but the goal canvas has no {missing}"
            ),
        ):
            plain.validate(frozenset({path, "masked_mimic.ref_pos"}), getup=False)
    submasked = GoalCanvas.derive(
        MaskedMimicControlConfig(anchor_pose_submasking=True), _robot()
    )
    with pytest.raises(ValueError, match=re.escape(SUBMASKING_UNBOUND)):
        submasked.validate(frozenset({"masked_mimic.ref_pos"}), getup=False)
    submasked.validate(frozenset({"masked_mimic.anchor_translation_mask"}), getup=False)
    full = GoalCanvas.derive(FULL, _robot())
    full.validate(
        frozenset(
            {
                "masked_mimic.local_ref_pos",
                "masked_mimic.style_pose_samples",
                "masked_mimic.velocity_command_mask",
                "masked_mimic.anchor_rotation_mask",
                "current.dof_pos",
            }
        ),
        getup=True,
    )


def test_configure_refuses_a_getup_preset_and_a_style_binding_the_checkpoint_lacks():
    """configure_masked_mimic_task checks the derived canvas against the bound
    observations and the get-up preset before building the task config."""

    def env_with(observations, control):
        return SimpleNamespace(
            control_components={"masked_mimic": control},
            observation_components=observations,
            termination_components={},
        )

    submasked = MaskedMimicControlConfig(anchor_pose_submasking=True)
    anchor_only = {"anchor": _binding(EnvContext.masked_mimic.anchor_rotation_mask)}
    with pytest.raises(ValueError, match=re.escape(GETUP_NEEDS_LOCAL_STREAM)):
        configure_masked_mimic_task(
            env_with(anchor_only, submasked),
            _robot(),
            "idle",
            getup=GetupRecoveryConfig(),
        )
    with_style = {
        **anchor_only,
        "style": _binding(EnvContext.masked_mimic.style_frame_masks),
    }
    with pytest.raises(ValueError, match="has no style stream"):
        configure_masked_mimic_task(env_with(with_style, submasked), _robot(), "idle")
    env = env_with(anchor_only, submasked)
    configure_masked_mimic_task(env, _robot(), "idle")
    task = env.control_components["masked_mimic"]
    assert task.canvas == GoalCanvas.derive(submasked, _robot())
    assert task.future_steps == (1,)


def test_bound_paths_unions_every_table():
    """bound_paths is the set of context paths bound across the given tables."""
    observations = {"a": _binding(EnvContext.current.dof_pos)}
    rewards = {
        "b": _binding(EnvContext.masked_mimic.ref_pos),
        "c": _binding(EnvContext.current.dof_pos),
    }
    assert bound_paths(observations, rewards, None) == frozenset(
        {"current.dof_pos", "masked_mimic.ref_pos"}
    )


def test_task_config_needs_its_canvas_and_the_controller_checks_it():
    """A task config without a canvas is refused; the task control's
    check_bindings validates the canvas with the get-up preset in mind."""
    from protomotions.envs.control.masked_mimic_tasks.controller import (
        MaskedMimicTaskControlConfig,
    )

    with pytest.raises(ValueError, match="needs its GoalCanvas"):
        MaskedMimicTaskControlConfig().validate()
    canvas = GoalCanvas.derive(
        MaskedMimicControlConfig(anchor_pose_submasking=True), _robot()
    )
    config = MaskedMimicTaskControlConfig(canvas=canvas, getup=GetupRecoveryConfig())
    with pytest.raises(ValueError, match=re.escape(GETUP_NEEDS_LOCAL_STREAM)):
        MaskedMimicTaskControl.check_bindings(
            config, _robot(), frozenset({"masked_mimic.anchor_rotation_mask"})
        )
    MaskedMimicTaskControl.check_bindings(
        MaskedMimicTaskControlConfig(canvas=canvas),
        _robot(),
        frozenset({"masked_mimic.anchor_rotation_mask"}),
    )
    # The training sampler's check derives the canvas itself.
    with pytest.raises(ValueError, match=re.escape(SUBMASKING_UNBOUND)):
        MaskedMimicControl.check_bindings(
            MaskedMimicControlConfig(anchor_pose_submasking=True), _robot(), frozenset()
        )


def test_local_stream_is_the_training_configs_switch():
    """The ``local_stream`` tag reads the training config's property, which is
    the one place the local probability is turned into a switch."""
    assert not MaskedMimicControlConfig().local_stream
    assert MaskedMimicControlConfig(local_target_condition_probability=0.1).local_stream
    assert MaskedMimicControl.provides(
        MaskedMimicControlConfig(local_target_condition_probability=0.1), "local_stream"
    )
    assert not MaskedMimicControl.provides(MaskedMimicControlConfig(), "local_stream")
    assert MaskedMimicTaskControl.provides(MaskedMimicControlConfig(), "local_stream")


@dataclass
class RefusingControlConfig(ControlComponentConfig):
    _target_: str = "protomotions.tests.test_goal_canvas.RefusingControl"


class RefusingControl(ControlComponent):
    """A control that refuses one binding, to show the build asks it."""

    def advance(self, state):
        return {}

    def publish(self, env_ids):
        return {}

    @classmethod
    def check_bindings(cls, config, robot_config, bound_paths):
        if "current.dof_pos" in bound_paths:
            raise ValueError("refused by the test control: current.dof_pos is bound")


def test_the_build_asks_every_control_about_the_bound_paths():
    """build.validate_bindings calls each configured control's check_bindings
    with the union of the three tables' bound paths, so a control's refusal
    fails the build with its own sentence."""
    pytest.importorskip("mujoco")
    from protomotions.tests.support.sensor_simulator import make_sensor_test_env

    with pytest.raises(
        ValueError, match="refused by the test control: current.dof_pos is bound"
    ):
        make_sensor_test_env(
            control_components={"refusing": RefusingControlConfig()},
            rewards={
                "dof": _binding(EnvContext.current.dof_pos, meta=RewardMeta(weight=1.0))
            },
        )
    env = make_sensor_test_env(control_components={"refusing": RefusingControlConfig()})
    try:
        assert isinstance(env.control_manager.components["refusing"], RefusingControl)
    finally:
        env.close()
