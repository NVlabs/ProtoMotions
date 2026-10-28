# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The simulator-free sample context does not drift from a real environment.

``deployment/sample_context.py`` rebuilds a deployment context by calling the
same publishers the environment calls, over synthetic inputs. This test is its
drift guard: it builds the parity harness's ``fake3_mimic`` environment -- the
real ``BaseEnv`` on the CPU sensor test double, with a sensor suite, a history
ring and a mimic control -- and asserts that every context name present in both
the real ``env.context`` and the stub has the same shape once the batch is set
to one. A new optional ``BodyState`` field, a renamed publisher, a wrong slot
count, or any shape the stub invents instead of reading off a publisher would
break here while the export tests, which share the stub, kept passing.

The stub reference leaves the optional ``RobotState`` fields (``dof_forces``,
``rigid_body_contacts``, ``local_rigid_body_rot``) ``None`` exactly as the motion
library does, and the second test pins that every name the stub publishes is one
the vocabulary declares (the scoped vocabulary check) -- which is what fails if a
reference double fabricates an undeclared ``mimic.ref_state.*`` field.
"""

import pytest


def _fake3_mimic_env():
    from protomotions.tests.parity.scenarios import SCENARIOS, build_env

    # The real BaseEnv, built the way train_agent builds it, on the CPU test
    # double -- no GPU, no real simulator. Its reset populates ``env.context``.
    return build_env(SCENARIOS["fake3_mimic"]).env


def _stub_for(env):
    from deployment.sample_context import build_sample_context

    # Size the stub's history ring from the same observation components the
    # environment sizes its own ring from, so the slot axes line up.
    context, _future, _slot = build_sample_context(
        env.robot_config,
        env.config,
        dict(env.config.observation_components),
        device="cpu",
    )
    return context


def test_stub_shapes_match_the_real_env_on_shared_paths():
    """Every path in both the real context and the stub agrees in shape and dtype.

    The exporter now writes each input's dtype off the stub's published tensor,
    so the dtype is checked here too (free). The batch axis aside, a shape or a
    dtype the stub invents instead of reading off a publisher would break here.
    """
    import torch

    env = _fake3_mimic_env()
    real = env.context
    stub = _stub_for(env)

    shared = sorted(set(real.keys()) & set(stub.keys()))
    # The two share the whole state and mimic surface; a build that shared only a
    # handful would mean the stub stopped publishing a namespace.
    assert len(shared) > 100, f"only {len(shared)} shared paths: {shared}"

    mismatches = []
    for path in shared:
        real_value, stub_value = real[path], stub[path]
        if torch.is_tensor(real_value) and torch.is_tensor(stub_value):
            if tuple(real_value.shape[1:]) != tuple(stub_value.shape[1:]):
                mismatches.append(
                    (path, tuple(real_value.shape), tuple(stub_value.shape))
                )
            elif real_value.dtype != stub_value.dtype:
                mismatches.append((path, real_value.dtype, stub_value.dtype))
        elif torch.is_tensor(real_value) != torch.is_tensor(stub_value):
            mismatches.append((path, "tensor-ness differs"))
        elif real_value != stub_value:
            mismatches.append((path, real_value, stub_value))
    assert not mismatches, f"shape/dtype drift on shared paths: {mismatches}"


def test_stub_publishes_only_declared_names():
    """Every name the stub publishes is one the vocabulary declares.

    Scoped by construction to the names the stub produces (the state and mimic
    namespaces and their top-level action rings); a reference double that set an
    optional ``RobotState`` field would publish an undeclared
    ``mimic.ref_state.*`` name and fail here.
    """
    from protomotions.envs.context_views import NAMESPACES, declared_paths

    env = _fake3_mimic_env()
    stub = _stub_for(env)
    declared = set()
    for namespace in NAMESPACES:
        declared |= set(declared_paths(namespace).keys())
    undeclared = sorted(set(stub.keys()) - declared)
    assert not undeclared, f"stub publishes undeclared names: {undeclared}"


# ---------------------------------------------------------------------------
# Masked-mimic (Metatron) prompt: a MaskedMimicControl publishes both mimic.*
# and masked_mimic.* in one simulator-free publish. These pin the masked_mimic.*
# half against the native metatron_v1 graph-input shapes
# (deployment/native/include/protomotions/deployment/metatron_v1_contract.hpp, on the
# feat/deploy branch at commit c0e5b49e2).
# ---------------------------------------------------------------------------

# Native metatron_v1 masked_mimic.* graph inputs and their shapes at batch 1.
_MASKED_MIMIC_SHAPES = {
    "masked_mimic.ref_pos": (1, 5, 33, 3),
    "masked_mimic.ref_rot": (1, 5, 33, 4),
    "masked_mimic.target_bodies_masks": (1, 50),
    "masked_mimic.target_poses_masks": (1, 5),
    "masked_mimic.time_offsets": (1, 5),
    "masked_mimic.anchor_translation_mask": (1, 5, 2),
    "masked_mimic.anchor_rotation_mask": (1, 5, 2),
    "masked_mimic.local_ref_pos": (1, 5, 33, 3),
    "masked_mimic.local_ref_rot": (1, 5, 33, 4),
    "masked_mimic.local_target_bodies_masks": (1, 50),
    "masked_mimic.local_target_poses_masks": (1, 5),
    "masked_mimic.local_time_offsets": (1, 5),
    "masked_mimic.style_pose_samples": (1, 225),
    "masked_mimic.style_velocity_samples": (1, 150),
    "masked_mimic.style_condition_masks": (1, 5, 11),
    "masked_mimic.style_frame_masks": (1, 5),
    "masked_mimic.average_velocity": (1, 2),
    "masked_mimic.velocity_command_mask": (1, 1),
}

# Every masked_mimic.* leaf published with all four modalities on: the 18 graph
# inputs above plus the two bare, non-graph-input target-time names.
_ALL_ON_MASKED_MIMIC = {p.split(".", 1)[1] for p in _MASKED_MIMIC_SHAPES} | {
    "target_times",
    "local_target_times",
}


def _g1_masked_mimic_env(**overrides):
    """A G1 masked-mimic config with all four modalities on (the metatron shape).

    ``overrides`` flip a modality switch for the drop test. Not a full training
    recipe: just the robot, the sensor suite and the masked_mimic control the
    sample publish reads.
    """
    from types import SimpleNamespace

    from examples.experiments.masked_mimic.deployable_g1_observations import (
        configure_robot,
    )
    from protomotions.envs.control.masked_mimic_control import (
        MaskedMimicControlConfig,
        MaskedMimicStyleConfig,
        MaskedMimicVelocityCommandConfig,
    )
    from protomotions.envs.sensors.config import (
        IMUSensor,
        JointEncoderSensor,
        LocalizationSensor,
        SensorConfig,
    )
    from protomotions.robot_configs.factory import robot_config

    robot = robot_config("g1")
    configure_robot(robot)  # the five trackable bodies -> C = 5
    config = dict(
        future_steps=[1, 2, 4, 8],
        num_masked_future_steps=5,
        local_target_condition_probability=0.5,
        anchor_pose_submasking=True,
        style=MaskedMimicStyleConfig(),
        velocity_command=MaskedMimicVelocityCommandConfig(horizon_seconds=0.5),
    )
    config.update(overrides)
    sensors = SensorConfig(
        joint_encoders=JointEncoderSensor(pos_noise=0.01, vel_noise=0.01),
        imu=IMUSensor(quat_noise=0.01, gyro_noise=0.01),
        localization=LocalizationSensor(
            body_name=robot.kinematic_info.body_names[0], noise=0.01
        ),
    )
    env = SimpleNamespace(
        sensors=sensors,
        control_components={"masked_mimic": MaskedMimicControlConfig(**config)},
    )
    return robot, env


def _masked_mimic_stub(robot, env):
    from deployment.sample_context import build_sample_context
    from protomotions.envs.control.masked_mimic_prompt import GoalCanvas
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.humanoid_historical import (
        compute_historical_actions_from_state,
    )
    from protomotions.envs.obs.masked_mimic import masked_mimic_goal_observations

    actor_obs = {
        "prev": MdpComponent.from_kernel(
            compute_historical_actions_from_state,
            history_steps=3,
            historical_actions=EnvContext.processed_action_history,
        )
    }
    canvas = GoalCanvas.derive(env.control_components["masked_mimic"], robot)
    actor_obs.update(masked_mimic_goal_observations(canvas))
    return build_sample_context(robot, env, actor_obs, device="cpu")


def test_masked_mimic_sample_context_matches_the_native_contract_shapes():
    """The simulator-free masked-mimic publish matches the native metatron_v1
    graph-input shapes exactly, across all five modality groups, so the stub
    cannot drift from the deployed tensor shapes."""
    import torch

    robot, env = _g1_masked_mimic_env()
    context, future_names, _slot = _masked_mimic_stub(robot, env)

    for path, shape in _MASKED_MIMIC_SHAPES.items():
        assert path in context, f"{path} not published"
        assert tuple(context[path].shape) == shape, (path, tuple(context[path].shape))
        expected = (
            torch.bool
            if path.endswith("masks") or path.endswith("mask")
            else torch.float32
        )
        assert context[path].dtype == expected, (path, context[path].dtype)
        assert torch.isfinite(context[path].float()).all(), path

    # A MaskedMimicControl also publishes the inherited mimic.* stream.
    assert any(k.startswith("mimic.") for k in context)

    # Body masks flatten the future/body/property axes but still depend on the
    # goal horizon. Style and velocity have independent, fixed-size layouts.
    mm_future = {n for n in future_names if n.startswith("masked_mimic.")}
    for name in (
        "ref_pos",
        "local_ref_pos",
        "anchor_translation_mask",
        "time_offsets",
        "target_bodies_masks",
        "local_target_bodies_masks",
    ):
        assert f"masked_mimic.{name}" in mm_future, name
    for flat in (
        "style_pose_samples",
        "style_condition_masks",
        "average_velocity",
    ):
        assert f"masked_mimic.{flat}" not in mm_future, flat


@pytest.mark.parametrize(
    "override, dropped",
    [
        (
            {"anchor_pose_submasking": False},
            {"anchor_translation_mask", "anchor_rotation_mask"},
        ),
        (
            {"local_target_condition_probability": 0.0},
            {
                "local_ref_pos",
                "local_ref_rot",
                "local_target_times",
                "local_time_offsets",
                "local_target_poses_masks",
                "local_target_bodies_masks",
            },
        ),
        (
            {"style": None},
            {
                "style_pose_samples",
                "style_velocity_samples",
                "style_condition_masks",
                "style_frame_masks",
            },
        ),
        (
            {"velocity_command": None},
            {"average_velocity", "velocity_command_mask"},
        ),
    ],
)
def test_switched_off_modality_drops_its_masked_mimic_inputs(override, dropped):
    """A disabled modality publishes none of its masked_mimic.* names.

    The host contract must not carry an input the policy never consumes, so with
    the config switch off the sample context drops exactly that group; the
    exporter would then refuse any actor that bound one.
    """
    robot, env = _g1_masked_mimic_env(**override)
    context, _future, _slot = _masked_mimic_stub(robot, env)
    present = {k.split(".", 1)[1] for k in context if k.startswith("masked_mimic.")}
    # Exactly the one group drops; everything else stays. Equality (not just
    # disjoint) also catches an over-drop or an accidental extra.
    assert present == _ALL_ON_MASKED_MIMIC - dropped, (override, present)
