# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo backend: actuator gear, integrator choice and root quaternion order.

SMPL, SMPL-X, SOMA23 and AMP declare their motors with a gear (50-500). MuJoCo
applies gear * actuator force to the joint, while ProtoMotions writes joint
torques and PD targets straight into ctrl, so the backend must reset the gear
to 1. These tests build the real simulator from the shipped assets.
"""

import numpy as np
import pytest
import torch

mujoco = pytest.importorskip("mujoco")

from protomotions.components.scene_lib import SceneLib  # noqa: E402
from protomotions.robot_configs.base import ControlType  # noqa: E402
from protomotions.robot_configs.factory import robot_config  # noqa: E402
from protomotions.simulator.base_simulator.simulator_state import (  # noqa: E402
    ResetState,
    StateConversion,
)
from protomotions.simulator.mujoco.config import MujocoSimulatorConfig  # noqa: E402
from protomotions.simulator.mujoco.simulator import MujocoSimulator  # noqa: E402

GEARED_ROBOTS = ["smpl", "soma23"]


def _build(rc, config):
    sim = MujocoSimulator(
        config=config,
        robot_config=rc,
        terrain=None,
        device=torch.device("cpu"),
        scene_lib=SceneLib.empty(num_envs=1, device="cpu", terrain=None),
    )
    sim._initialize_with_markers({})
    return sim


def _make_sim(robot, control_type=None, use_implicit_pd=True, integrator="keep"):
    rc = robot_config(robot)
    if control_type is not None:
        rc.control.control_type = control_type
    sim_params = rc.simulation_params.mujoco
    if integrator != "keep":
        sim_params.integrator = integrator
    config = MujocoSimulatorConfig(
        num_envs=1,
        headless=True,
        experiment_name="test_mujoco_actuators",
        sim=sim_params,
        use_implicit_pd=use_implicit_pd,
    )
    return _build(rc, config)


def _actuated_kp(sim):
    """Stiffness of each actuated DOF in qpos order, read from the robot config."""
    n = sim._num_actuated_dofs
    names = [
        mujoco.mj_id2name(
            sim.model, mujoco.mjtObj.mjOBJ_JOINT, sim.model.dof_jntid[6 + i]
        )
        for i in range(n)
    ]
    info = sim.robot_config.control.control_info
    return np.array([info[name].stiffness for name in names])


def _set_dof_pos(sim, dof_pos):
    n = sim._num_actuated_dofs
    sim.data.qpos[7 : 7 + n] = dof_pos
    sim.data.qvel[:] = 0.0
    mujoco.mj_forward(sim.model, sim.data)


def _random_targets(n, seed=0):
    # float32 round trip so qpos holds exactly what ctrl received
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.3, 0.3, n).astype(np.float32).astype(np.float64)


@pytest.mark.parametrize("robot", GEARED_ROBOTS)
def test_position_actuators_hold_the_target_itself(robot):
    sim = _make_sim(robot)
    n = sim._num_actuated_dofs
    assert np.all(sim.model.actuator_gear[:, 0] == 1.0)

    target = _random_targets(n)
    sim._apply_simulator_pd_targets(torch.tensor(target, dtype=torch.float32)[None])

    # Before the fix this was gear * effort (up to 250,000 N*m), pushing away.
    _set_dof_pos(sim, target)
    np.testing.assert_allclose(sim.data.qfrc_actuator[6 : 6 + n], 0.0, atol=1e-3)

    delta = 0.01
    _set_dof_pos(sim, target + delta)
    np.testing.assert_allclose(
        sim.data.qfrc_actuator[6 : 6 + n], -_actuated_kp(sim) * delta, rtol=1e-4
    )


@pytest.mark.parametrize("robot", GEARED_ROBOTS)
def test_torque_mode_applies_the_commanded_joint_torque(robot):
    sim = _make_sim(robot, control_type=ControlType.TORQUE)
    n = sim._num_actuated_dofs

    torques = np.linspace(-5.0, 5.0, n).astype(np.float32)
    sim._apply_simulator_torques(torch.tensor(torques)[None])
    mujoco.mj_forward(sim.model, sim.data)

    np.testing.assert_allclose(sim.data.qfrc_actuator[6 : 6 + n], torques, atol=1e-5)


@pytest.mark.parametrize("robot", GEARED_ROBOTS)
def test_explicit_pd_applies_the_torque_it_reports(robot):
    sim = _make_sim(robot, use_implicit_pd=False)
    n = sim._num_actuated_dofs

    target = _random_targets(n)
    delta = 0.01
    _set_dof_pos(sim, target + delta)
    sim._apply_simulator_pd_targets(torch.tensor(target, dtype=torch.float32)[None])
    mujoco.mj_forward(sim.model, sim.data)

    applied = sim.data.qfrc_actuator[6 : 6 + n]
    np.testing.assert_allclose(applied, sim._last_applied_torques, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(applied, -_actuated_kp(sim) * delta, rtol=1e-4)


def test_gear_reset_keeps_declared_limits_in_joint_units():
    xml = """
    <mujoco>
      <worldbody>
        <body>
          <joint name="j" type="hinge" axis="1 0 0"/>
          <geom size="0.1"/>
        </body>
      </worldbody>
      <actuator>
        <motor joint="j" gear="50" ctrllimited="true" ctrlrange="-1 1"
               forcelimited="true" forcerange="-2 2"/>
      </actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    def joint_torque(ctrl):
        data.ctrl[0] = ctrl
        mujoco.mj_forward(model, data)
        return data.qfrc_actuator[0]

    commands = [0.5, 1.0, 3.0, -4.0]
    before = [joint_torque(c) for c in commands]

    sim = object.__new__(MujocoSimulator)
    sim.model = model
    sim._neutralize_actuator_gear()

    assert model.actuator_gear[0, 0] == 1.0
    # ctrl is now the joint torque; the MJCF's +-50 N*m limit survives
    after = [joint_torque(50.0 * c) for c in commands]
    np.testing.assert_allclose(after, before)
    assert max(abs(t) for t in after) == pytest.approx(50.0)


def _gear_reset_on(joint_type, gear):
    """Run the gear reset on a single motor driving one joint of `joint_type`."""
    xml = f"""
    <mujoco>
      <worldbody>
        <body>
          <joint name="j" type="{joint_type}"/>
          <geom size="0.1"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="m" joint="j" gear="{gear}"/>
      </actuator>
    </mujoco>
    """
    sim = object.__new__(MujocoSimulator)
    sim.model = mujoco.MjModel.from_xml_string(xml)
    sim._neutralize_actuator_gear()
    return sim.model


@pytest.mark.parametrize("joint_type", ["hinge", "slide"])
def test_gear_reset_accepts_a_positive_gear_on_a_one_dof_joint(joint_type):
    model = _gear_reset_on(joint_type, "50")
    np.testing.assert_array_equal(model.actuator_gear[0], [1, 0, 0, 0, 0, 0])


@pytest.mark.parametrize("gear", ["0", "-50"])
def test_gear_reset_rejects_a_zero_or_negative_gear(gear):
    with pytest.raises(ValueError, match="needs a positive gear"):
        _gear_reset_on("hinge", gear)


def test_gear_reset_rejects_a_ball_joint_actuator():
    # A ball joint's gear is a 3D torque axis; collapsing it to a scalar 1
    # would silently drive the joint about the wrong axis.
    with pytest.raises(ValueError, match="hinge and slide joints only"):
        _gear_reset_on("ball", "0 1 0 0 0 0")


def test_reset_after_switching_from_isaaclab_writes_wxyz():
    from protomotions.simulator.factory import update_simulator_config_for_test
    from protomotions.simulator.isaaclab.config import IsaacLabSimulatorConfig

    rc = robot_config("soma23")
    config = IsaacLabSimulatorConfig(
        num_envs=1,
        headless=True,
        experiment_name="test_mujoco_actuators",
        sim=rc.simulation_params.isaaclab,
    )
    assert config.w_last is True
    config = update_simulator_config_for_test(config, "mujoco", rc)
    assert config.w_last is False
    sim = _build(rc, config)
    n = sim._num_actuated_dofs

    # Turned ~145 degrees, mostly about the vertical axis
    root_rot = torch.tensor([[0.1, -0.2, 0.9, 0.3]])
    root_rot = root_rot / root_rot.norm(dim=-1, keepdim=True)
    state = ResetState(
        root_pos=torch.tensor([[0.0, 0.0, 1.0]]),
        root_rot=root_rot.clone(),
        root_vel=torch.zeros(1, 3),
        root_ang_vel=torch.zeros(1, 3),
        dof_pos=torch.zeros(1, n),
        dof_vel=torch.zeros(1, n),
        state_conversion=StateConversion.COMMON,
    )
    sim.reset_envs(state)

    np.testing.assert_allclose(
        sim.data.qpos[3:7], root_rot[0, [3, 0, 1, 2]].numpy(), atol=1e-6
    )
    read_back = sim.get_robot_state().root_rot[0]
    assert torch.dot(read_back, root_rot[0]).abs().item() == pytest.approx(
        1.0, abs=1e-6
    )


@pytest.mark.parametrize(
    "integrator, expected",
    [
        ("keep", mujoco.mjtIntegrator.mjINT_IMPLICITFAST),  # smpl's robot config
        (None, mujoco.mjtIntegrator.mjINT_EULER),  # MJCF / MuJoCo default
        ("rk4", mujoco.mjtIntegrator.mjINT_RK4),
    ],
)
def test_integrator_comes_from_sim_params(integrator, expected):
    sim = _make_sim("smpl", integrator=integrator)
    assert sim.model.opt.integrator == expected


def test_unknown_integrator_is_rejected():
    with pytest.raises(ValueError, match="Unknown MuJoCo integrator"):
        _make_sim("smpl", integrator="verlet")
