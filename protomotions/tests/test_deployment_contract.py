# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The deployment contract validates itself and round-trips through the sidecar."""

from dataclasses import replace

import pytest
import yaml

from deployment.contract import (
    SCHEMA,
    ControlBlock,
    DeploymentContract,
    HistorySlotSpec,
    InputSpec,
    MotionBlock,
    OutputSpec,
    RobotBlock,
    RuntimeBlock,
    TimingBlock,
    provider_of,
)
from protomotions.envs.sensors.metadata import IMUMetadata, SensorDeploymentMetadata
from protomotions.envs.control.masked_mimic_prompt import GoalCanvas


def _joint_input(**overrides) -> InputSpec:
    base = dict(
        onnx_name="current_dof_pos",
        key="current.dof_pos",
        source_namespace="current",
        kind="joint_pos",
        provider="simulator_truth",
        shape=(1, 2),
        dtype="float32",
        element_names=(("j1", "j2"),),
        future_steps=None,
        history=None,
    )
    base.update(overrides)
    return InputSpec(**base)


def _outputs():
    return (
        OutputSpec("actions", "actions", (1, 2), None),
        OutputSpec("joint_pos_targets", "joint_pos_targets", (1, 2), ("j1", "j2")),
    )


def _robot() -> RobotBlock:
    return RobotBlock(
        mjcf_path="robot.xml",
        body_names=("pelvis", "head"),
        joint_names=("j1", "j2"),
        anchor_body_name="pelvis",
        root_body_name="pelvis",
    )


def _control() -> ControlBlock:
    return ControlBlock(
        action_fn="bm_pd_action",
        stiffness=(100.0, 100.0),
        damping=(10.0, 10.0),
        effort_limits=(50.0, 50.0),
        action_scale=(0.25, 0.25),
        pd_action_offset=(0.0, 0.0),
        pd_target_max_accel=None,
    )


def _goal_canvas() -> GoalCanvas:
    return GoalCanvas(
        future_steps=(1, 2),
        num_masked_future_steps=2,
        conditionable_bodies=("head", "pelvis"),
        anchor_body="pelvis",
        anchor_pose_submasking=False,
        has_local_stream=False,
        style_frames=None,
        has_velocity_command=False,
    )


def _runtime() -> RuntimeBlock:
    return RuntimeBlock(
        onnx_in_names=("current_dof_pos",),
        onnx_out_names=("actions", "joint_pos_targets"),
        onnx_name_to_in_key={"current_dof_pos": "current.dof_pos"},
        passthrough_keys=(),
        obs_context_keys=("current.dof_pos",),
    )


def _contract(**overrides) -> DeploymentContract:
    base = dict(
        schema=SCHEMA,
        inputs=(_joint_input(),),
        outputs=_outputs(),
        robot=_robot(),
        control=_control(),
        timing=TimingBlock(control_dt=0.02, physics_dt=0.005),
        motion=MotionBlock(future_step_indices=(1, 2)),
        runtime=_runtime(),
        sensors=None,
        checkpoint="last.ckpt",
        control_type="BUILT_IN_PD",
        onnx_sha256=None,
    )
    base.update(overrides)
    return DeploymentContract(**base)


def test_valid_contract_builds_and_derives_indices():
    contract = _contract()
    assert contract.robot.anchor_body_index == 0
    assert contract.robot.root_body_index == 0
    assert contract.timing.decimation == 4
    assert contract.motion.future_dt_seconds(0.02) == [0.02, 0.04]


def test_to_yaml_writes_schema_first_and_keeps_todays_layout(tmp_path):
    path = _contract().to_yaml(tmp_path / "unified_pipeline.yaml")
    with open(path) as handle:
        content = yaml.safe_load(handle)

    assert list(content)[0] == "schema"
    assert content["schema"] == SCHEMA
    assert list(content) == [
        "schema",
        "type",
        "dt",
        "joint_names",
        "body_names",
        "default_joint_stiffness",
        "default_joint_damping",
        "policy_inputs",
        "policy_outputs",
        "_runtime",
        "metadata",
        "sensors",
        "robot",
        "control",
        "timing",
        "motion",
    ]
    assert content["type"] == "unified_pipeline"
    assert set(content["control"]) == {
        "stiffness",
        "damping",
        "effort_limits",
        "action_scale",
        "pd_action_offset",
        "pd_target_max_accel",
        "action_fn",
        "action_ema_alpha",
    }
    # Kept at today's constant so the sidecar stays a strict superset of the
    # file the host reads today; the host-side filter removal is after handover.
    assert content["control"]["action_ema_alpha"] == 1.0
    assert content["control"]["action_fn"] == "bm_pd_action"
    assert content["robot"]["anchor_body_index"] == 0


def test_round_trip_through_yaml_preserves_blocks(tmp_path):
    sensors = SensorDeploymentMetadata(
        imu=IMUMetadata(
            body_name="pelvis",
            body_index=0,
            mount_sensor_to_body_xyzw=[0.0, 0.0, 0.0, 1.0],
        ),
        anchor_body_name="pelvis",
    )
    history_input = _joint_input(
        onnx_name="raw_action_history",
        key="raw_action_history",
        source_namespace="raw_action_history",
        kind="last_actions",
        provider="policy_history",
        shape=(1, 3, 2),
        element_names=(),
        history=HistorySlotSpec(
            output_key="actions",
            slots=3,
            first_action_lag=1,
            reset_fill="zeros",
            steps_back=(1, 2, 3),
            reset_value=0.0,
        ),
        source="the graph's actions output, slot i from steps_back[i] calls ago",
    )
    original = _contract(
        inputs=(
            _joint_input(source="joint position in radians", dtype="float64"),
            history_input,
        ),
        sensors=sensors,
        onnx_sha256="abc123",
    )
    path = original.to_yaml(tmp_path / "unified_pipeline.yaml")
    restored = DeploymentContract.from_yaml(path)

    assert restored.schema == SCHEMA
    assert restored.robot == original.robot
    assert restored.control == original.control
    assert restored.timing == original.timing
    assert restored.motion == original.motion
    assert restored.sensors == original.sensors
    assert restored.onnx_sha256 == "abc123"
    assert restored.inputs[0].source == "joint position in radians"
    # dtype survives the round trip instead of silently resetting to float32.
    assert restored.inputs[0].dtype == "float64"
    assert restored.inputs[1].history == history_input.history


def test_per_input_layout_is_a_strict_superset_of_todays_keys(tmp_path):
    # Every key the pre-contract hand-built YAML emitted per input must still be
    # present, so a host that reads the file today reads the new file unchanged
    # (the board's strict-superset rule until the after-handover schema bump).
    history_input = _joint_input(
        onnx_name="raw_action_history",
        key="raw_action_history",
        source_namespace="raw_action_history",
        kind="last_actions",
        provider="policy_history",
        shape=(1, 3, 2),
        element_names=(),
        history=HistorySlotSpec(
            output_key="actions",
            slots=3,
            first_action_lag=1,
            reset_fill="zeros",
            steps_back=(1, 2, 3),
            reset_value=0.0,
        ),
        source="last actions, before any deploy-side clamp",
    )
    future_input = _joint_input(
        onnx_name="mimic_future_pos",
        key="mimic.future_pos",
        source_namespace="mimic",
        kind="reference_motion_body_pos_future",
        provider="reference_motion",
        shape=(1, 2, 2, 3),
        element_names=(("pelvis", "head"), ("x", "y", "z")),
        future_steps=2,
        source="reference body positions at each future step",
    )
    contract = _contract(inputs=(history_input, future_input))
    path = contract.to_yaml(tmp_path / "unified_pipeline.yaml")
    with open(path) as handle:
        content = yaml.safe_load(handle)

    by_name = {entry["name"]: entry for entry in content["policy_inputs"]}
    history = by_name["raw_action_history"]
    for key in (
        "name",
        "key",
        "shape",
        "kind",
        "output_key",
        "history",
        "first_action_lag",
        "steps_back",
        "fill_before_available",
        "source",
        "reset_value",
    ):
        assert key in history, f"history input lost today's key {key!r}"

    future = by_name["mimic_future_pos"]
    for key in (
        "name",
        "key",
        "shape",
        "kind",
        "element_names",
        "future_steps",
        "source",
    ):
        assert key in future, f"regular input lost today's key {key!r}"


def test_from_yaml_refuses_a_foreign_schema(tmp_path):
    path = tmp_path / "unified_pipeline.yaml"
    path.write_text(yaml.dump({"schema": "unified_pipeline/9", "policy_inputs": []}))
    with pytest.raises(ValueError, match="is not 'unified_pipeline/1'"):
        DeploymentContract.from_yaml(path)


def test_input_without_kind_or_provider_is_refused():
    with pytest.raises(ValueError, match="no deployment kind or provider"):
        _contract(inputs=(_joint_input(kind=""),))
    with pytest.raises(ValueError, match="no deployment kind or provider"):
        _contract(inputs=(_joint_input(provider=""),))


def test_history_input_feeding_a_missing_output_is_refused():
    bad = _joint_input(
        kind="last_actions",
        history=HistorySlotSpec(
            output_key="not_an_output", slots=2, first_action_lag=1, reset_fill="zeros"
        ),
    )
    with pytest.raises(ValueError, match="which the graph does not produce"):
        _contract(inputs=(bad,))


def test_anchor_not_in_body_list_is_refused():
    robot = RobotBlock(
        mjcf_path="robot.xml",
        body_names=("pelvis", "head"),
        joint_names=("j1", "j2"),
        anchor_body_name="left_foot",
        root_body_name="pelvis",
    )
    with pytest.raises(ValueError, match="is not in the robot's body list"):
        _contract(robot=robot)


def test_control_vector_length_mismatch_is_refused():
    with pytest.raises(ValueError, match="control.stiffness has 1 entries"):
        _contract(
            control=ControlBlock(
                action_fn="bm_pd_action",
                stiffness=(100.0,),
                damping=(10.0, 10.0),
                effort_limits=(50.0, 50.0),
                action_scale=None,
                pd_action_offset=None,
                pd_target_max_accel=None,
            )
        )


def test_missing_effort_limit_is_refused():
    with pytest.raises(ValueError, match="effort_limits must be set"):
        _contract(
            control=ControlBlock(
                action_fn="bm_pd_action",
                stiffness=(100.0, 100.0),
                damping=(10.0, 10.0),
                effort_limits=None,
                action_scale=None,
                pd_action_offset=None,
                pd_target_max_accel=None,
            )
        )


def test_sensor_input_without_its_channel_is_refused():
    loc_input = _joint_input(
        key="noisy.root_pos",
        source_namespace="noisy",
        kind="localization_pos",
        provider="localization",
    )
    # sensors is None: no channel carries the localization input.
    with pytest.raises(ValueError, match="which the sensors block does not carry"):
        _contract(inputs=(loc_input,))


def test_duplicate_onnx_input_name_is_refused():
    with pytest.raises(ValueError, match="duplicate ONNX input name"):
        _contract(inputs=(_joint_input(), _joint_input()))


def test_provider_of_dispatches_on_namespace():
    # A sensor path is supplied by the channel the caller derives; the other
    # namespaces ignore the channel argument entirely.
    assert provider_of("noisy.projected_gravity", "imu") == "imu"
    assert provider_of("clean_historical.dof_pos", "imu") == "imu"
    assert provider_of("current.dof_pos", None) == "simulator_truth"
    assert provider_of("historical.dof_pos", None) == "simulator_truth"
    assert provider_of("mimic.ref_anchor_pos", None) == "reference_motion"
    assert provider_of("raw_action_history", None) == "policy_history"


def test_provider_of_refuses_a_sensor_path_with_no_channel():
    # A sensor path whose name derives from no sensor-measured field has no
    # channel, so it cannot be fed and is refused.
    with pytest.raises(ValueError, match="no sensor channel"):
        provider_of("noisy.root_pos", None)
    # A privileged path ignores the channel: simulator truth needs none.
    assert provider_of("current.dof_pos", None) == "simulator_truth"


def test_robot_block_conditionable_body_names_round_trips_and_is_masked_only(tmp_path):
    # Plain tracker: the field is None and the sidecar omits the key, so a plain
    # export stays byte-identical.
    plain_path = _contract().to_yaml(tmp_path / "plain.yaml")
    plain_doc = yaml.safe_load(plain_path.read_text())
    assert "conditionable_body_names" not in plain_doc["robot"]
    restored_plain = DeploymentContract.from_yaml(plain_path)
    assert restored_plain.robot.conditionable_body_names is None

    # Masked export: the list is written, in publisher order, and round-trips as
    # a tuple (the host indexes its own body/style masks by this order).
    masked_robot = RobotBlock(
        mjcf_path="robot.xml",
        body_names=("pelvis", "head"),
        joint_names=("j1", "j2"),
        anchor_body_name="pelvis",
        root_body_name="pelvis",
        conditionable_body_names=("head", "pelvis"),
    )
    masked_path = _contract(robot=masked_robot, masked_mimic=_goal_canvas()).to_yaml(
        tmp_path / "masked.yaml"
    )
    masked_doc = yaml.safe_load(masked_path.read_text())
    assert masked_doc["robot"]["conditionable_body_names"] == ["head", "pelvis"]
    restored = DeploymentContract.from_yaml(masked_path)
    assert restored.robot.conditionable_body_names == ("head", "pelvis")
    assert restored.masked_mimic.conditionable_bodies == ("head", "pelvis")


@pytest.mark.parametrize("change", ["reorder", "missing_robot_bodies", "missing_goals"])
def test_sidecar_refuses_inconsistent_conditionable_bodies(tmp_path, change):
    goals = _goal_canvas()
    robot = replace(_robot(), conditionable_body_names=goals.conditionable_bodies)
    content = _contract(robot=robot, masked_mimic=goals).to_dict()
    if change == "reorder":
        content["robot"]["conditionable_body_names"].reverse()
    elif change == "missing_robot_bodies":
        del content["robot"]["conditionable_body_names"]
    else:
        del content["masked_mimic"]
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(content))
    with pytest.raises(ValueError, match="conditionable_body_names must match"):
        DeploymentContract.from_yaml(path)
