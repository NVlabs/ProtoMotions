# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sensor/FK contracts shared by training and deployment input packing."""

import pickle
import io
from types import SimpleNamespace

import pytest
import torch

from examples.experiments.masked_mimic.deployable_g1_observations import (
    deployable_proprioception_observations,
)
from protomotions.envs.base_env.publishers import prefix_names
from protomotions.envs.context_sources import sensed
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.obs.state_history_buffer import history_depth
from protomotions.envs.state_names import body_state_names
from protomotions.envs.component_manager import ComponentManager
from protomotions.envs.obs.humanoid_historical import compute_historical_proprioception
from functools import partial
from protomotions.envs.obs.masked_mimic import compute_target_poses_only
from protomotions.envs.sensors.config import (
    IMUSensor,
    JointEncoderSensor,
    LocalizationSensor,
    NoNoise,
    SensorConfig,
)
from protomotions.tests.support.sensor_simulator import make_sensor_test_env
from protomotions.envs.obs.state_history_buffer import HistorySample, StateHistoryBuffer
from protomotions.envs.state import BodyState
from protomotions.tests import test_baked_fk
from protomotions.utils.baked_fk import BakedAnchorFK


@pytest.fixture
def branching_ki():
    return test_baked_fk.branching_ki.__wrapped__()


def identity(*shape):
    value = torch.zeros(*shape, 4)
    value[..., 3] = 1
    return value


def state(batch=2):
    return BodyState(
        rigid_body_pos=torch.zeros(batch, 5, 3), rigid_body_rot=identity(batch, 5),
        rigid_body_vel=torch.zeros(batch, 5, 3), rigid_body_ang_vel=torch.zeros(batch, 5, 3),
        dof_pos=torch.zeros(batch, 4), dof_vel=torch.zeros(batch, 4),
        rigid_body_contacts=torch.zeros(batch, 5, dtype=torch.bool),
    )


def history():
    s = state()
    return StateHistoryBuffer(8, HistorySample({"noisy": s}, torch.zeros(2, 4), torch.zeros(2, 4)))


def test_numbered_action_trace_excludes_current_slot(branching_ki):
    buffer = history()
    s = state()
    for t in range(12):
        s.dof_pos.fill_(t)
        buffer.push(HistorySample({"noisy": s}, torch.full((2, 4), float(t - 1)), torch.zeros(2, 4)))
    captured = buffer.capture().map(lambda value: value[:, 1:])
    view = body_state_names(captured.states["noisy"], 1)
    fk = BakedAnchorFK(branching_ki, 1, torch.arange(5))
    tokens = compute_historical_proprioception(
        view["dof_pos"],
        view["dof_vel"],
        view["root_local_ang_vel"],
        view["anchor_rot"],
        captured.raw_actions,
        0.02,
        fk=fk,
        history_steps=[8, 4, 3, 2, 1],
    ).reshape(2, 5, -1)
    # s(11-k) carries a(10-k); processed actions are never substituted.
    torch.testing.assert_close(tokens[0, :, -5:-1], torch.tensor([2., 6., 7., 8., 9.])[:, None].expand(-1, 4))
    torch.testing.assert_close(tokens[0, :, -1], torch.tensor([8, 4, 3, 2, 1]) * 0.02)


@pytest.mark.parametrize("channel", ["joint_pos", "imu_quat", "imu_gyro"])
def test_deployable_pack_retains_one_sensor_frame_through_reset_and_replay(channel):
    exact = NoNoise("isolate one sensor")
    levels = {"joint_pos": exact, "joint_vel": exact, "imu_quat": exact, "imu_gyro": exact}
    levels[channel] = 0.2
    sensors = SensorConfig(
        joint_encoders=JointEncoderSensor(levels["joint_pos"], levels["joint_vel"]),
        imu=IMUSensor(levels["imu_quat"], levels["imu_gyro"]),
        localization=LocalizationSensor("pelvis", exact),
    )
    env = make_sensor_test_env(sensors=sensors, history_steps=8)
    try:
        components = deployable_proprioception_observations(
            env.robot_config.kinematic_info, torch.arange(4), 1,
        )
        manager = ComponentManager(torch.device("cpu"))
        def pack():
            return manager.execute_all(components, env.context)
        first = pack()
        rng = env.sensors.generator.get_state().clone()
        for name, value in pack().items():
            torch.testing.assert_close(value, first[name], rtol=0, atol=0)
        assert torch.equal(rng, env.sensors.generator.get_state())
        env.step(torch.full((3, 2), .3))
        saved_ctx = env.context
        expected_action = saved_ctx["current_action"].clone()
        snapshot = env.save_state()
        expected = {name: value.clone() for name, value in pack().items()}
        env.reset(torch.tensor([1]))
        reset = pack()
        for name in expected:
            torch.testing.assert_close(reset[name][[0, 2]], expected[name][[0, 2]], rtol=0, atol=0)
        assert torch.count_nonzero(reset["last_action"][1]) == 0
        tokens = reset["historical_proprioception"][1].reshape(5, -1)
        torch.testing.assert_close(tokens[:, :-1], tokens[:1, :-1].expand(5, -1), rtol=0, atol=0)
        env.restore_state(snapshot)
        for name, value in pack().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        env.step(torch.ones(3, 2))
        torch.testing.assert_close(
            saved_ctx["current_action"], expected_action, rtol=0, atol=0
        )
    finally:
        env.close()


def test_sparse_goals_are_independently_grounded_and_masked(branching_ki):
    pos, rot = torch.rand(2, 5, 3), identity(2, 5)
    target = partial(compute_target_poses_only, torch.arange(5),
                     anchor_body_index=1, future_steps=[1, 3],
                     include_root_relative=False, ground_independently=True)
    refs = pos[:, None].expand(-1, 3, -1, -1).clone()
    ref_rots = rot[:, None].expand(-1, 3, -1, -1)
    masks = torch.ones(2, 30, dtype=torch.bool)
    before = target(pos, rot, refs, ref_rots, masks)
    refs[..., 2] += 13
    shifted_pos = pos + torch.tensor([0., 0., -7.])
    after = target(shifted_pos, rot, refs, ref_rots, masks)
    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)
    hidden = target(pos, rot, refs, ref_rots, torch.zeros_like(masks))
    assert torch.count_nonzero(hidden) == 0


def test_training_and_deployment_proprioception_pack_parity():
    env = make_sensor_test_env(history_steps=8)
    try:
        ctx = env.context
        components = deployable_proprioception_observations(
            env.robot_config.kinematic_info, torch.arange(4), 1,
        )
        restored = pickle.loads(pickle.dumps(components))
        manager = ComponentManager(torch.device("cpu"))
        packed = manager.execute_all(restored, ctx)
        h = lambda name: ctx[f"noisy_historical.{name}"]
        expected = components["historical_proprioception"].compute_func(
            h("dof_pos"),
            h("dof_vel"),
            h("root_local_ang_vel"),
            h("anchor_rot"),
            h("raw_actions"),
            env.dt,
            **components["historical_proprioception"].static_params,
        )
        torch.testing.assert_close(packed["historical_proprioception"], expected)
        torch.testing.assert_close(packed["last_action"], ctx["current_action"])
        # Retained sensor readings, not mutable privileged tensors, feed the pack.
        env.simulator.state.rigid_body_pos.fill_(1000)
        env.simulator.state.rigid_body_vel.fill_(999)
        torch.testing.assert_close(manager.execute_all(components, ctx)["fk_max_coords_obs"],
                                   packed["fk_max_coords_obs"])
        goal = MdpComponent.from_kernel(
            compute_target_poses_only, conditionable_body_ids=torch.arange(4)
        )
        goal = sensed({"goal": goal}, "noisy")["goal"]
        assert goal.dynamic_vars["current_state_body_pos"] == "noisy.rigid_body_pos"
    finally:
        env.close()


@pytest.mark.parametrize("anchor_submasks", [False, True])
def test_g1_wiring_shares_body_order_and_preserves_privileged_keys(branching_ki, anchor_submasks):
    from examples.experiments.masked_mimic.deployable_g1_observations import configure_observations, configure_robot

    robot = SimpleNamespace(
        kinematic_info=branching_ki, anchor_body_index=1, trackable_bodies_subset=[],
        imu_body_name=branching_ki.body_names[0], imu_mount_quat=(0., 0., 0., 1.),
        common_naming_to_robot_body_names=dict(zip(
            ["torso_body_name", "all_left_hand_bodies", "all_right_hand_bodies",
             "all_left_foot_bodies", "all_right_foot_bodies"],
            [[name] for name in branching_ki.body_names],
        )),
    )
    from protomotions.robot_configs.base import RobotConfig

    robot.body_indices = lambda *names: RobotConfig.body_indices(robot, *names)
    env = SimpleNamespace(
        observation_components={"privileged": object()},
        control_components={"masked_mimic": SimpleNamespace(anchor_pose_submasking=anchor_submasks)},
    )
    configure_robot(robot)
    configure_observations(robot, env)
    assert "privileged" in env.observation_components
    # The deployable observations read eight past slots; the ring sizes itself.
    added = {k: v for k, v in env.observation_components.items() if k != "privileged"}
    assert history_depth(added) == 8
    goal = env.observation_components["fk_masked_mimic_target_poses"]
    assert goal.static_params["conditionable_body_ids"].tolist() == list(range(5))
    assert goal.static_params.get("future_steps") is None
    assert ("anchor_translation_mask" in goal.dynamic_vars) == anchor_submasks


@pytest.mark.parametrize("steps", [[], [0], [-1], [1, 1], [1.5], [True]])
def test_invalid_history_lags_rejected(branching_ki, steps):
    # The component normalises the step selection at construction: an empty,
    # non-positive or repeated step is a ValueError, a non-int a TypeError.
    with pytest.raises((TypeError, ValueError), match="history_steps"):
        deployable_proprioception_observations(branching_ki, torch.arange(5), 1, steps)


@pytest.mark.parametrize("kind", ["history", "goals"])
def test_new_observation_kernels_export_and_run_onnx(branching_ki, kind):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")

    class Export(torch.nn.Module):
        def __init__(self, kernel):
            super().__init__()
            self.kernel = kernel

        def forward(self, *values):
            if kind == "history":
                return compute_historical_proprioception(
                    *values, dt=0.02, fk=self.kernel, history_steps=[8, 4, 3, 2, 1],
                )
            return self.kernel(*values)

    fk = BakedAnchorFK(branching_ki, 1, torch.arange(5))
    if kind == "history":
        kernel = fk
        values = (torch.rand(2, 8, 4), torch.rand(2, 8, 4), torch.rand(2, 8, 3),
                  identity(2, 8), torch.rand(2, 8, 4))
    else:
        kernel = partial(compute_target_poses_only, torch.arange(5),
                         anchor_body_index=1, future_steps=[1, 3],
                         include_root_relative=False, ground_independently=True)
        pos, body_rot = torch.rand(2, 5, 3), identity(2, 5)
        values = (pos, body_rot, pos[:, None].expand(-1, 3, -1, -1).clone(),
                  body_rot[:, None].expand(-1, 3, -1, -1).clone(),
                  torch.ones(2, 30, dtype=torch.bool))
    module = Export(kernel)
    stream = io.BytesIO()
    names = [f"input_{i}" for i in range(len(values))]
    torch.onnx.export(module, values, stream, input_names=names, output_names=["observation"],
                      dynamic_axes={name: {0: "batch"} for name in [*names, "observation"]},
                      opset_version=17, dynamo=False)
    session = ort.InferenceSession(stream.getvalue(), providers=["CPUExecutionProvider"])
    for batch in (1, 2, 3):
        inputs = tuple(value[:1].expand(batch, *value.shape[1:]).clone() for value in values)
        output = session.run(None, {name: value.numpy() for name, value in zip(names, inputs)})[0]
        torch.testing.assert_close(torch.from_numpy(output), module(*inputs), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("source", ["clean", "noisy"])
def test_global_sensor_goals_require_localization(source):
    from protomotions.tests.test_sensor_target_bindings import sensor_context
    ctx = sensor_context()
    ctx["mimic.anchor_idx"] = 1
    ctx.update(
        {
            "masked_mimic.ref_pos": torch.zeros(2, 3, 3, 3),
            "masked_mimic.ref_rot": identity(2, 3, 3),
            "masked_mimic.target_bodies_masks": torch.ones(2, 18, dtype=torch.bool),
        }
    )
    goal = MdpComponent.from_kernel(
        compute_target_poses_only,
        conditionable_body_ids=torch.arange(3),
        ground_independently=True,
    )
    goal = sensed({"goal": goal}, source)["goal"]
    manager = ComponentManager(torch.device("cpu"))
    with pytest.raises(ValueError, match=source + r".rigid_body_pos"):
        manager.validate_bindings({"goal": goal}, ctx, group="observation")

    measured = BodyState(
        rigid_body_pos=torch.rand(2, 3, 3), rigid_body_rot=identity(2, 3),
        rigid_body_ang_vel=torch.zeros(2, 3, 3),
        dof_pos=torch.zeros(2, 2), dof_vel=torch.zeros(2, 2),
    )
    ctx.update(prefix_names(source, body_state_names(measured, 1)))
    manager.validate_bindings({"goal": goal}, ctx, group="observation")
    assert torch.isfinite(goal.compute(ctx)).all()
