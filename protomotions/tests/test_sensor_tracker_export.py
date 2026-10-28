# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sensor source bindings must survive export without changing host semantics."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deployment.tracker_inputs import (
    build_tracker_onnx_inputs,
    tracker_input_keys,
    tracker_sensor_metadata,
)
from protomotions.envs.context_sources import sensed
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.obs.humanoid import compute_humanoid_reduced_coords_observations
from protomotions.envs.sensors.config import (
    IMUSensor,
    JointEncoderSensor,
    NoNoise,
    SensorConfig,
)
from protomotions.utils.export_utils import ObservationExportModule


def _reduced(source="privileged"):
    """The reduced-coords kernel, repointed at ``source``."""
    component = MdpComponent.from_kernel(compute_humanoid_reduced_coords_observations)
    return sensed({"self": component}, source)["self"]


def _exact():
    """A mandatory-channel-only SensorConfig measured exactly (no optional channels)."""
    exact = NoNoise("export test")
    return SensorConfig(
        joint_encoders=JointEncoderSensor(exact, exact), imu=IMUSensor(exact, exact)
    )


@pytest.mark.parametrize("source", ["clean", "noisy"])
def test_tracker_wire_names_leave_source_bindings_and_history_unchanged(source):
    keys = [f"{source}.dof_pos", f"{source}_historical.dof_pos"]
    assert tracker_input_keys(keys) == ["current.dof_pos", keys[1]]
    assert keys[0] == f"{source}.dof_pos"
    with pytest.raises(ValueError, match="collides"):
        tracker_input_keys([f"{source}.dof_pos", "current.dof_pos"])


def test_metadata_is_only_emitted_for_sensor_backed_actor():
    from protomotions.tests.support.sensor_simulator import SensorTestSimulator

    robot = SensorTestSimulator().robot_config
    env = SimpleNamespace(
        sensors=_exact(),
        observation_components={"self": _reduced("noisy")},
    )
    metadata = tracker_sensor_metadata(env, robot, ["self"])
    assert metadata.imu.gyro_frame == "sensor"
    assert metadata.history_initialization == "repeat_first_reading"
    env.sensors = None
    with pytest.raises(ValueError, match="self.*noisy.*without env.sensors"):
        tracker_sensor_metadata(env, robot, ["self"])
    env.observation_components["self"] = _reduced()
    assert tracker_sensor_metadata(env, robot, ["self"]) is None


def _contract(onnx_keys, num_dofs=2, anchor_idx=0, num_bodies=1):
    """A minimal valid DeploymentContract whose inputs are ``onnx_keys``.

    The reference-host input assembler reads only ``.inputs`` and ``.robot``, so
    the other blocks carry the smallest values the contract's validation accepts.
    """
    from deployment.contract import (
        SCHEMA,
        ControlBlock,
        DeploymentContract,
        InputSpec,
        MotionBlock,
        OutputSpec,
        RobotBlock,
        RuntimeBlock,
        TimingBlock,
    )

    joint_names = tuple(f"j{i}" for i in range(num_dofs))
    body_names = tuple(f"b{i}" for i in range(num_bodies))
    inputs = tuple(
        InputSpec(
            onnx_name=name,
            key=key,
            source_namespace=key.split(".", 1)[0],
            kind="x",
            provider="simulator_truth",
            shape=(1, num_dofs),
            dtype="float32",
            element_names=(),
            future_steps=None,
            history=None,
        )
        for name, key in onnx_keys.items()
    )
    return DeploymentContract(
        schema=SCHEMA,
        inputs=inputs,
        outputs=(OutputSpec("actions", "actions", (1, num_dofs), None),),
        robot=RobotBlock(
            "robot.xml", body_names, joint_names, body_names[anchor_idx], body_names[0]
        ),
        control=ControlBlock(
            "fn",
            (1.0,) * num_dofs,
            (0.1,) * num_dofs,
            (10.0,) * num_dofs,
            None,
            None,
            None,
        ),
        timing=TimingBlock(0.02, 0.005),
        motion=MotionBlock((1, 2)),
        runtime=RuntimeBlock(
            tuple(onnx_keys),
            ("actions",),
            dict(onnx_keys),
            (),
            tuple(onnx_keys.values()),
        ),
        sensors=None,
        checkpoint="c.ckpt",
        control_type="BUILT_IN_PD",
        onnx_sha256=None,
    )


def test_host_action_history_keeps_policy_outputs_and_joint_targets_distinct():
    supplied = build_tracker_onnx_inputs(
        _contract(
            {
                "raw": "raw_action_history",
                "processed": "processed_action_history",
            },
            num_dofs=2,
        ),
        dof_pos=np.zeros(2, dtype=np.float32),
        dof_vel=np.zeros(2, dtype=np.float32),
        anchor_rot=np.array([0, 0, 0, 1], dtype=np.float32),
        root_local_ang_vel=np.zeros(3, dtype=np.float32),
        future_refs={},
        prev_raw_actions=np.array([2.0, -3.0], dtype=np.float32),
        prev_actions=np.array([0.2, -0.1], dtype=np.float32),
    )
    np.testing.assert_array_equal(supplied["raw"], [[[2.0, -3.0]]])
    np.testing.assert_allclose(supplied["processed"], [[[0.2, -0.1]]])


@pytest.mark.parametrize("source", ["clean", "noisy"])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_real_onnx_observations_accept_host_sensor_inputs(tmp_path, source, batch_size):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")

    # A tracing input per bound sensor path. (The exporter proper reads these off
    # the simulator-free sample context; a single reduced-coords observation over
    # a stand-in robot has no sensor runtime or mimic control, so the two bound
    # paths are materialised directly by their leaf.)
    component = _reduced(source)
    _leaf_shape = {
        "dof_pos": (1, 2),
        "dof_vel": (1, 2),
        "anchor_rot": (1, 4),
        "root_local_ang_vel": (1, 3),
    }
    samples = {}
    for path in component.get_bindings_dict().values():
        leaf = path.rsplit(".", 1)[-1]
        tensor = torch.randn(*_leaf_shape[leaf])
        if leaf == "anchor_rot":
            tensor = torch.nn.functional.normalize(tensor, dim=-1)
        samples[path] = tensor
    module = ObservationExportModule({"self": component}, samples, torch.device("cpu"))
    source_keys = module.get_input_keys()
    wire_keys = tracker_input_keys(source_keys)
    names = [key.replace(".", "_") for key in wire_keys]
    path = tmp_path / "sensor.onnx"
    torch.onnx.export(
        module,
        tuple(samples[key] for key in source_keys),
        str(path),
        input_names=names,
        output_names=["self"],
        dynamic_axes={name: {0: "batch"} for name in [*names, "self"]},
        opset_version=17,
        dynamo=False,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    mapping = dict(zip(names, wire_keys))
    supplied = build_tracker_onnx_inputs(
        _contract(mapping, num_dofs=2, anchor_idx=1, num_bodies=2),
        dof_pos=np.array([0.2, -0.3], dtype=np.float32),
        dof_vel=np.array([1.0, -2.0], dtype=np.float32),
        anchor_rot=np.array([0.0, 0.0, 0.6, 0.8], dtype=np.float32),
        root_local_ang_vel=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        future_refs={},
    )
    supplied = {
        name: np.repeat(value, batch_size, axis=0) for name, value in supplied.items()
    }
    expected = module(*(torch.from_numpy(supplied[name]) for name in names))[0]
    actual = session.run(None, supplied)[0]
    np.testing.assert_allclose(actual, expected.numpy(), atol=1e-6)
