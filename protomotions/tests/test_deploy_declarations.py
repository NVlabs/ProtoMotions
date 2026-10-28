# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one ``deploy=`` host kind on a context name derives the exporter's values.

Each tracker observation name carries a single ``deploy=`` string, the host's
name for the quantity. Everything the hand-written exporter ladder used to spell
out -- the column labels, the host provider and the tensor shape -- is now
derived: the labels from the kind-keyed ``element_labels`` table, the provider
from the namespace plus the sensor channel the name derives from, and the shape
from a simulator-free publish. This test resolves each name against a g1
sample context and pins those derived values against the same ``CASES`` table
the typed-fact ladder was checked against.
"""

from types import SimpleNamespace

import pytest

from deployment.contract import element_labels, provider_for_path
from protomotions.envs.context_views import deploy_for

# The host labels an exporter writes use the robot's own joint and body names;
# these stand-ins let the label assertions read like the old typed ladder.
JOINTS = ("j1", "j2")
BODIES = ("pelvis", "head")
ANCHOR = "pelvis"


# (path, kind string, host provider, element labels, g1 sample-context shape).
# The shapes are g1's own (29 dofs, 33 bodies, a 4-step future horizon, a
# depth-3 history ring); the labels use the stand-in names above.
CASES = [
    ("current.dof_pos", "joint_pos", "simulator_truth", (JOINTS,), (1, 29)),
    ("noisy.dof_pos", "joint_pos", "joint_encoders", (JOINTS,), (1, 29)),
    ("current.dof_vel", "joint_vel", "simulator_truth", (JOINTS,), (1, 29)),
    (
        "current.anchor_rot",
        "anchor_rot",
        "simulator_truth",
        (("x", "y", "z", "w"),),
        (1, 4),
    ),
    ("current.root_local_ang_vel", "local_root_ang_vel", "simulator_truth", (), (1, 3)),
    ("current.anchor_pos", "anchor_pos", "simulator_truth", (("x", "y", "z"),), (1, 3)),
    ("noisy.anchor_pos", "anchor_pos", "localization", (("x", "y", "z"),), (1, 3)),
    ("noisy.odom_start_xy", "odom_start_xy", "localization", (("x", "y"),), (1, 2)),
    ("noisy.odom_disp_start", "odom_disp_start", "localization", (("x", "y"),), (1, 2)),
    (
        "noisy.odom_start_heading_inv",
        "odom_start_heading_inv",
        "localization",
        (("x", "y", "z", "w"),),
        (1, 4),
    ),
    (
        "mimic.ref_anchor_pos",
        "reference_motion_anchor_pos",
        "reference_motion",
        ((ANCHOR,), ("x", "y", "z")),
        (1, 3),
    ),
    (
        "mimic.ref_state.rigid_body_pos",
        "reference_motion_body_pos",
        "reference_motion",
        (BODIES, ("x", "y", "z")),
        (1, 33, 3),
    ),
    (
        "mimic.future_anchor_pos",
        "reference_motion_anchor_pos_future",
        "reference_motion",
        ((ANCHOR,), ("x", "y", "z")),
        (1, 4, 3),
    ),
    (
        "mimic.future_anchor_rot",
        "reference_motion_anchor_rot",
        "reference_motion",
        ((ANCHOR,), ("x", "y", "z", "w")),
        (1, 4, 4),
    ),
    (
        "mimic.future_anchor_vel",
        "reference_motion_anchor_vel",
        "reference_motion",
        ((ANCHOR,), ("x", "y", "z")),
        (1, 4, 3),
    ),
    (
        "mimic.future_anchor_ang_vel",
        "reference_motion_anchor_ang_vel",
        "reference_motion",
        ((ANCHOR,), ("x", "y", "z")),
        (1, 4, 3),
    ),
    (
        "mimic.future_pos",
        "reference_motion_body_pos_future",
        "reference_motion",
        (BODIES, ("x", "y", "z")),
        (1, 4, 33, 3),
    ),
    (
        "mimic.future_rot",
        "reference_motion_body_rot",
        "reference_motion",
        (BODIES, ("x", "y", "z", "w")),
        (1, 4, 33, 4),
    ),
    (
        "mimic.future_dof_pos",
        "reference_motion_joint_pos",
        "reference_motion",
        (JOINTS,),
        (1, 4, 29),
    ),
    (
        "mimic.future_dof_vel",
        "reference_motion_joint_vel",
        "reference_motion",
        (JOINTS,),
        (1, 4, 29),
    ),
    # The policy's own past actions, fed back as the tracker's history input.
    ("raw_action_history", "last_actions", "policy_history", (JOINTS,), (1, 3, 29)),
    (
        "processed_action_history",
        "last_actions",
        "policy_history",
        (JOINTS,),
        (1, 3, 29),
    ),
]


def _full_sensor_stub():
    """One g1 sample context with every sensor channel, a history ring and mimic.

    All five channels are declared so the clean/noisy sources carry every
    estimated field, a reach-3 action-history observation sizes the ring so the
    history axes appear, and the mimic horizon publishes the future windows. The
    context then publishes every declared deployable name, so the derivation is
    exercised over all of them in one build.
    """
    from deployment.sample_context import build_sample_context
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.humanoid_historical import (
        compute_historical_actions_from_state,
    )
    from protomotions.envs.sensors.config import (
        BaseVelocitySensor,
        HeightSensor,
        IMUSensor,
        JointEncoderSensor,
        LocalizationSensor,
        SensorConfig,
    )
    from protomotions.robot_configs.factory import robot_config

    robot = robot_config("g1")
    sensors = SensorConfig(
        joint_encoders=JointEncoderSensor(pos_noise=0.01, vel_noise=0.01),
        imu=IMUSensor(quat_noise=0.01, gyro_noise=0.01),
        localization=LocalizationSensor(
            body_name=robot.kinematic_info.body_names[0], noise=0.01
        ),
        base_velocity=BaseVelocitySensor(noise=0.01),
        height=HeightSensor(noise=0.01),
    )
    env = SimpleNamespace(
        sensors=sensors,
        control_components={"mimic": MimicControlConfig(future_steps=[1, 2, 4, 8])},
    )
    actor_obs = {
        "prev": MdpComponent.from_kernel(
            compute_historical_actions_from_state,
            history_steps=3,
            historical_actions=EnvContext.processed_action_history,
        )
    }
    context, _future, _slot = build_sample_context(robot, env, actor_obs, device="cpu")
    return context


def test_each_tracker_name_derives_like_todays_ladder():
    context = _full_sensor_stub()
    for path, kind, provider, element_names, shape in CASES:
        assert deploy_for(path) == kind, path
        assert path in context, f"{path} not published by the stub"
        width = context[path].shape[-1]
        assert (
            element_labels(kind, JOINTS, BODIES, ANCHOR, width) == element_names
        ), path
        assert provider_for_path(path) == provider, path
        assert tuple(context[path].shape) == shape, (path, tuple(context[path].shape))


def test_command_history_is_policy_history_in_any_namespace():
    # The kept processed-action command history reads policy_history under each
    # historical namespace, the same host kind the top-level action rings carry.
    for namespace in ("historical", "clean_historical", "noisy_historical"):
        path = f"{namespace}.processed_actions"
        assert deploy_for(path) == "last_actions", path
        assert provider_for_path(path) == "policy_history", path


def test_sensor_history_exports_but_privileged_history_is_refused():
    # The host replays sensor measurements, not privileged simulator states.
    import torch

    from deployment.export_to_onnx import _build_samples
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent

    assert deploy_for("noisy_historical.dof_pos") == "joint_pos"
    assert deploy_for("historical.dof_pos") == "joint_pos"

    def _reads(x):
        return x

    component = MdpComponent(_reads, {"x": EnvContext.noisy_historical.dof_pos})
    context = {"noisy_historical.dof_pos": torch.zeros(1, 5, 29)}
    samples = _build_samples({"hist": component}, context, {"noisy_historical.dof_pos"})
    from deployment.input_history import history_specs

    action, measurement = history_specs(
        "noisy_historical.dof_pos", samples["noisy_historical.dof_pos"].shape
    )
    assert action is None
    assert measurement.current_key == "noisy.dof_pos"
    assert measurement.steps_back == (1, 2, 3, 4, 5)
    assert measurement.reset_fill == "initial_measurement"
    # Privileged simulator-truth history stays refused.
    priv = MdpComponent(_reads, {"x": EnvContext.historical.dof_pos})
    ctx = {"historical.dof_pos": torch.zeros(1, 8, 29)}
    with pytest.raises(ValueError, match="history axis the host cannot feed"):
        _build_samples({"p": priv}, ctx, {"historical.dof_pos"})


def test_sensor_history_input_round_trips_without_an_output_key():
    # Measurements declare their current input and lag, not a graph output.
    from deployment.contract import (
        InputSpec,
        ObservationHistorySpec,
        _input_dict,
        _input_from_dict,
    )

    spec = InputSpec(
        onnx_name="noisy_historical_dof_pos",
        key="noisy_historical.dof_pos",
        source_namespace="noisy_historical",
        kind="joint_pos",
        provider="joint_encoders",
        shape=(1, 8, 29),
        dtype="float32",
        element_names=(),
        future_steps=None,
        history=None,
        observation_history=ObservationHistorySpec(
            current_key="noisy.dof_pos",
            steps_back=tuple(range(1, 9)),
            reset_fill="initial_measurement",
        ),
    )
    entry = _input_dict(spec)
    assert "output_key" not in entry
    assert "history" not in entry
    assert entry["observation_history"] == {
        "current_key": "noisy.dof_pos",
        "steps_back": list(range(1, 9)),
        "reset_fill": "initial_measurement",
    }

    back = _input_from_dict(entry)
    assert back.history is None
    assert back.observation_history == spec.observation_history


def test_unknown_kind_is_refused_by_the_label_table():
    # A misspelled deploy kind is a refusal, listing the known kinds.
    with pytest.raises(ValueError, match="no element-label rule"):
        element_labels("joint_position", ("a",), ("b",), "b", 3)


# The full MaskedMimic (Metatron) operator-prompt interface, the 18 masked_mimic.*
# graph inputs the native metatron_v1 contract carries, in five groups. Each
# declaration carries a per-field masked_mimic_* kind (the native input name) and
# provider "masked_mimic". target_times / local_target_times are not graph inputs
# and stay bare. Raw pose tensors label the full body axis; masks/times/packed
# style carry no element labels. (path, kind, coord width, element labels.)
MASKED_MIMIC_CASES = [
    # Global targets.
    ("masked_mimic.ref_pos", "masked_mimic_ref_pos", 3, (BODIES, ("x", "y", "z"))),
    (
        "masked_mimic.ref_rot",
        "masked_mimic_ref_rot",
        4,
        (BODIES, ("x", "y", "z", "w")),
    ),
    ("masked_mimic.target_bodies_masks", "masked_mimic_target_bodies_masks", 2, ()),
    ("masked_mimic.target_poses_masks", "masked_mimic_target_poses_masks", 1, ()),
    ("masked_mimic.time_offsets", "masked_mimic_time_offsets", 1, ()),
    # Anchor submasks.
    (
        "masked_mimic.anchor_translation_mask",
        "masked_mimic_anchor_translation_mask",
        2,
        (),
    ),
    ("masked_mimic.anchor_rotation_mask", "masked_mimic_anchor_rotation_mask", 2, ()),
    # Local targets (raw poses the obs graph localises).
    (
        "masked_mimic.local_ref_pos",
        "masked_mimic_local_ref_pos",
        3,
        (BODIES, ("x", "y", "z")),
    ),
    (
        "masked_mimic.local_ref_rot",
        "masked_mimic_local_ref_rot",
        4,
        (BODIES, ("x", "y", "z", "w")),
    ),
    (
        "masked_mimic.local_target_bodies_masks",
        "masked_mimic_local_target_bodies_masks",
        2,
        (),
    ),
    (
        "masked_mimic.local_target_poses_masks",
        "masked_mimic_local_target_poses_masks",
        1,
        (),
    ),
    ("masked_mimic.local_time_offsets", "masked_mimic_local_time_offsets", 1, ()),
    # Style (already-packed feature arrays -> none).
    ("masked_mimic.style_pose_samples", "masked_mimic_style_pose_samples", 1, ()),
    (
        "masked_mimic.style_velocity_samples",
        "masked_mimic_style_velocity_samples",
        1,
        (),
    ),
    ("masked_mimic.style_condition_masks", "masked_mimic_style_condition_masks", 1, ()),
    ("masked_mimic.style_frame_masks", "masked_mimic_style_frame_masks", 1, ()),
    # Velocity command (planar [vx, vy]).
    (
        "masked_mimic.average_velocity",
        "masked_mimic_average_velocity",
        2,
        (("x", "y"),),
    ),
    (
        "masked_mimic.velocity_command_mask",
        "masked_mimic_velocity_command_mask",
        1,
        (),
    ),
]


def test_masked_mimic_prompt_names_resolve_kind_and_provider():
    # Each prompt name resolves its per-field host kind and the operator-prompt
    # provider, and its kind has a label row of the right shape (bodies for raw
    # poses, coords for the planar velocity, none for masks/times/packed style).
    for path, kind, width, element_names in MASKED_MIMIC_CASES:
        assert deploy_for(path) == kind, path
        assert provider_for_path(path) == "masked_mimic", path
        assert (
            element_labels(kind, JOINTS, BODIES, ANCHOR, width) == element_names
        ), path


def test_undeclared_masked_mimic_streams_are_refused_at_export():
    # target_times / local_target_times are not graph inputs, so they carry no
    # deploy= string and the exporter refuses a bound one rather than ship a
    # graph input no host can feed.
    import torch

    from deployment.export_to_onnx import _build_samples
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent

    assert deploy_for("masked_mimic.target_times") is None
    assert deploy_for("masked_mimic.local_target_times") is None

    def _reads(x):
        return x

    component = MdpComponent(_reads, {"x": EnvContext.masked_mimic.target_times})
    context = {"masked_mimic.target_times": torch.zeros(1, 4)}
    with pytest.raises(ValueError, match="no deployment declaration"):
        _build_samples({"g": component}, context, set())
