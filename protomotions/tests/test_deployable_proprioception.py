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
from protomotions.envs.obs.masked_mimic import (
    compute_fk_target_poses_only,
    compute_target_poses_only,
)
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
        # The deployable sibling reads joints + anchor pose and runs the FK the
        # proprioception already carries; body poses are no longer an input.
        fk = components["historical_proprioception"].static_params["fk"]
        deployable = MdpComponent.from_kernel(
            compute_fk_target_poses_only, fk=fk, conditionable_body_ids=torch.arange(4)
        )
        deployable = sensed({"goal": deployable}, "noisy")["goal"]
        assert deployable.dynamic_vars["dof_pos"] == "noisy.dof_pos"
        assert deployable.dynamic_vars["anchor_pos"] == "noisy.anchor_pos"
        assert deployable.dynamic_vars["anchor_rot"] == "noisy.anchor_rot"
        assert "current_state_body_pos" not in deployable.dynamic_vars
        assert "current_state_body_rot" not in deployable.dynamic_vars
        assert deployable.static_params["fk"] is fk
    finally:
        env.close()


def test_fk_kernel_world_poses_equal_the_sensor_runtime():
    """In-graph FK from the published readings reproduces the published poses.

    ``noisy.rigid_body_pos/rot`` are what the sensor runtime composed from the
    IMU, the localization and the joint FK; the deployable goal kernel recomputes
    them in the graph from ``noisy.dof_pos``, ``noisy.anchor_pos`` and
    ``noisy.anchor_rot`` with the same FK. The two must agree to float rounding
    on every step, and the deployable goal must equal the privileged kernel fed
    the sensor runtime's poses.
    """
    env = make_sensor_test_env(history_steps=8)
    try:
        anchor = env.robot_config.anchor_body_index
        bodies = torch.arange(env.robot_config.kinematic_info.num_bodies)
        fk = BakedAnchorFK(env.robot_config.kinematic_info, anchor, bodies)
        goal = partial(
            compute_target_poses_only, bodies, anchor_body_index=anchor,
            future_steps=[1, 3], include_root_relative=False, ground_independently=True,
        )
        fk_goal = partial(
            compute_fk_target_poses_only, bodies, anchor_body_index=anchor,
            future_steps=[1, 3], include_root_relative=False, ground_independently=True,
            fk=fk,
        )
        generator = torch.Generator().manual_seed(3)
        for _ in range(10):
            env.step(torch.rand(3, 2, generator=generator) - 0.5)
            ctx = env.context
            pos, rot = fk.world_body_poses(
                ctx["noisy.dof_pos"], ctx["noisy.anchor_pos"], ctx["noisy.anchor_rot"]
            )
            torch.testing.assert_close(pos, ctx["noisy.rigid_body_pos"], atol=1e-5, rtol=1e-5)
            dots = (rot * ctx["noisy.rigid_body_rot"]).sum(-1).abs()
            torch.testing.assert_close(dots, torch.ones_like(dots), atol=1e-5, rtol=0)
            refs = torch.rand(3, 3, len(bodies), 3, generator=generator)
            ref_rots = identity(3, 3, len(bodies))
            masks = torch.rand(3, 3 * 2 * len(bodies), generator=generator) > 0.3
            torch.testing.assert_close(
                fk_goal(ctx["noisy.dof_pos"], ctx["noisy.anchor_pos"],
                        ctx["noisy.anchor_rot"], refs, ref_rots, masks),
                goal(ctx["noisy.rigid_body_pos"], ctx["noisy.rigid_body_rot"],
                     refs, ref_rots, masks),
                atol=1e-5, rtol=1e-5,
            )
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
    # The goal runs the one FK kernel the recipe built, shared with the
    # proprioception, on the sensor readings; it binds no body poses.
    history = env.observation_components["historical_proprioception"]
    assert isinstance(goal.static_params["fk"], BakedAnchorFK)
    assert goal.static_params["fk"] is history.static_params["fk"]
    assert goal.static_params["fk"].anchor_idx == robot.anchor_body_index
    assert goal.dynamic_vars["dof_pos"] == "noisy.dof_pos"
    assert goal.dynamic_vars["anchor_pos"] == "noisy.anchor_pos"
    assert goal.dynamic_vars["anchor_rot"] == "noisy.anchor_rot"
    assert not {"current_state_body_pos", "current_state_body_rot"} & set(goal.dynamic_vars)
    # Without the flag the recipe binds no scene name.
    assert not any(
        str(path).startswith("scene_boxes.")
        for component in added.values()
        for path in component.dynamic_vars.values()
    )


def test_g1_wiring_binds_the_scene_pair_to_the_same_fk_kernel_when_asked(branching_ki):
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
        observation_components={},
        control_components={"masked_mimic": SimpleNamespace(anchor_pose_submasking=True)},
    )
    configure_robot(robot)
    configure_observations(robot, env, scene_boxes=True)
    corners = env.observation_components["scene_box_corners"]
    mask = env.observation_components["scene_box_mask"]
    goal = env.observation_components["fk_masked_mimic_target_poses"]
    # One kernel for the goal, the proprioception and the scene corners, so the
    # export folds one FK and every body pose comes from the same readings.
    assert corners.static_params["fk"] is goal.static_params["fk"]
    assert corners.dynamic_vars["dof_pos"] == "noisy.dof_pos"
    assert corners.dynamic_vars["centers"] == "scene_boxes.centers"
    assert mask.dynamic_vars == {"visible": "scene_boxes.visible", "valid": "scene_boxes.valid"}


def test_fk_exporter_detects_one_binding_by_its_constants_not_identity(branching_ki):
    """Two goals whose kernels were rebuilt from the bundle still count as one.

    The config bundle saves a kernel by its constants and rebuilds one object
    per component, so the exporter compares what the bundle saves (class, anchor,
    body subset) plus the bound paths; a kernel anchored elsewhere is refused.
    """
    from deployment.export_fk_to_onnx import fk_binding

    def goal(fk, ids):
        component = MdpComponent.from_kernel(
            compute_fk_target_poses_only, fk=fk, conditionable_body_ids=ids
        )
        return sensed({"goal": component}, "noisy")["goal"]

    same_a = BakedAnchorFK(branching_ki, 1, torch.arange(5))
    same_b = BakedAnchorFK(branching_ki, 1, torch.arange(5))
    assert same_a is not same_b
    fk, paths = fk_binding(
        {"first": goal(same_a, torch.arange(5)), "second": goal(same_b, torch.arange(3))}
    )
    # sensed() keeps the kernel object (its rebind copies the static params
    # shallowly), so identity would hold here; the exporter still compares what
    # the bundle saves, because a saved bundle rebuilds one kernel per
    # component. same_a and same_b are distinct objects with equal constants,
    # so that comparison is what this exercises.
    assert type(fk) is BakedAnchorFK and fk.export_init() == same_a.export_init()
    assert paths == ("noisy.dof_pos", "noisy.anchor_pos", "noisy.anchor_rot")
    assert fk_binding({"plain": MdpComponent.from_kernel(same_a)}) is None
    other_anchor = BakedAnchorFK(branching_ki, 2, torch.arange(5))
    with pytest.raises(ValueError, match="different FK kernels or state paths"):
        fk_binding(
            {"first": goal(same_a, torch.arange(5)), "second": goal(other_anchor, torch.arange(5))}
        )


@pytest.mark.parametrize("steps", [[], [0], [-1], [1, 1], [1.5], [True]])
def test_invalid_history_lags_rejected(branching_ki, steps):
    # The component normalises the step selection at construction: an empty,
    # non-positive or repeated step is a ValueError, a non-int a TypeError.
    with pytest.raises((TypeError, ValueError), match="history_steps"):
        deployable_proprioception_observations(branching_ki, torch.arange(5), 1, steps)


@pytest.mark.parametrize("kind", ["history", "goals", "fk_goals"])
def test_new_observation_kernels_export_and_run_onnx(branching_ki, kind):
    ort = pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")

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
    elif kind == "goals":
        kernel = partial(compute_target_poses_only, torch.arange(5),
                         anchor_body_index=1, future_steps=[1, 3],
                         include_root_relative=False, ground_independently=True)
        pos, body_rot = torch.rand(2, 5, 3), identity(2, 5)
        values = (pos, body_rot, pos[:, None].expand(-1, 3, -1, -1).clone(),
                  body_rot[:, None].expand(-1, 3, -1, -1).clone(),
                  torch.ones(2, 30, dtype=torch.bool))
    else:
        # The deployable sibling: joints + anchor pose in, FK inside the graph.
        kernel = partial(compute_fk_target_poses_only, torch.arange(5),
                         anchor_body_index=1, future_steps=[1, 3],
                         include_root_relative=False, ground_independently=True,
                         fk=fk)
        pos, body_rot = torch.rand(2, 5, 3), identity(2, 5)
        values = (torch.rand(2, 4), torch.rand(2, 3), identity(2),
                  pos[:, None].expand(-1, 3, -1, -1).clone(),
                  body_rot[:, None].expand(-1, 3, -1, -1).clone(),
                  torch.ones(2, 30, dtype=torch.bool))
    module = Export(kernel)
    stream = io.BytesIO()
    names = [f"input_{i}" for i in range(len(values))]
    torch.onnx.export(module, values, stream, input_names=names, output_names=["observation"],
                      dynamic_axes={name: {0: "batch"} for name in [*names, "observation"]},
                      opset_version=17, dynamo=False)
    if kind == "fk_goals":
        # export_fk_to_onnx traces the FK on its own for a static TensorRT
        # plan, so it must stay free of data-dependent ops.
        ops = {node.op_type for node in onnx.load_from_string(stream.getvalue()).graph.node}
        assert not ops & {"NonZero", "If", "Loop", "Scan"}, sorted(ops)
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
