# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo effort readback follows the last step through reset and evaluation."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

mujoco = pytest.importorskip("mujoco")

from protomotions.components.scene_lib import SceneLib  # noqa: E402
from protomotions.envs.base_env.owners import SimulatorPhysicalState  # noqa: E402
from protomotions.robot_configs.base import ControlType  # noqa: E402
from protomotions.robot_configs.factory import robot_config  # noqa: E402
from protomotions.simulator.base_simulator.simulator_state import (  # noqa: E402
    ResetState,
    StateConversion,
)
from protomotions.simulator.mujoco.config import MujocoSimulatorConfig  # noqa: E402
from protomotions.simulator.mujoco.simulator import MujocoSimulator  # noqa: E402


def _step_shell(model, floating=False, latency=0.0):
    sim = object.__new__(MujocoSimulator)
    sim.model, sim.data = model, mujoco.MjData(model)
    sim._has_free_joint = floating
    sim._num_actuated_dofs = 1
    sim._applied_dof_forces = np.zeros(1)
    sim._dof_to_actuator = np.array([0])
    sim._apply_torques_to_ctrl(np.array([999.0]))
    sim._poll_keyboard_events = lambda: None
    sim._apply_control = lambda: None
    sim._max_action_latency_ms = latency
    sim.config = SimpleNamespace(use_implicit_pd=True)
    sim.control_type = ControlType.BUILT_IN_PD
    sim.decimation = 4
    sim._step_count = 1
    return sim


@pytest.mark.parametrize("floating", [False, True])
def test_readback_uses_limited_joint_effort_and_excludes_other_forces(floating):
    """Readback adds applied joint force after actuator limits and robot slicing."""
    free = '<freejoint name="root"/>' if floating else ""
    model = mujoco.MjModel.from_xml_string(f"""<mujoco>
      <option gravity="0 0 0"/>
      <worldbody><body name="base" pos="0 0 2">{free}
        <geom size=".1" mass="1" contype="0" conaffinity="0"/>
        <body><joint name="joint" damping="3"/>
          <geom size=".1" mass="1" contype="0" conaffinity="0"/>
        </body></body>
        <body name="projectile" pos="4 0 2"><freejoint/>
          <geom size=".1" mass="1"/>
        </body></worldbody>
      <actuator><motor joint="joint" gear="2" forcelimited="true"
                       forcerange="-2 2"/></actuator>
    </mujoco>""")
    sim = _step_shell(model, floating)
    start = 6 if floating else 0
    sim.data.qvel[start] = 2
    sim.data.ctrl[0] = 100
    sim.data.qfrc_applied[start] = 0.75
    sim.data.qfrc_applied[start + 1 :] = 123
    if floating:
        sim.data.qfrc_applied[:6] = 456
    sim._physics_step()
    assert sim.data.actuator_force[0] == 2
    assert sim.data.qfrc_actuator[start] == 4
    assert sim.data.qfrc_passive[start] != 0
    before = (sim.data.time, sim.data.qpos.copy(), sim.data.qvel.copy())
    result = sim._get_simulator_dof_forces()
    assert result.state_conversion is StateConversion.SIMULATOR
    torch.testing.assert_close(result.dof_forces, torch.tensor([[4.75]]))
    assert sim.data.time == before[0]
    np.testing.assert_array_equal(sim.data.qpos, before[1])
    np.testing.assert_array_equal(sim.data.qvel, before[2])
    result.dof_forces.zero_()
    sim.data.qfrc_actuator[:] = 0
    sim.data.qfrc_applied[:] = 0
    torch.testing.assert_close(
        sim._get_simulator_dof_forces().dof_forces, torch.tensor([[4.75]])
    )


@pytest.mark.parametrize("latency", [0.0, 1.0])
def test_readback_keeps_final_substep_effort_before_forward(latency):
    """The final applied effort survives the next-state actuator evaluation."""
    model = mujoco.MjModel.from_xml_string("""<mujoco>
      <option gravity="0 0 0" timestep=".01" integrator="implicitfast"/>
      <worldbody><body><joint name="j"/><geom size=".1" mass="1"/>
      </body></worldbody>
      <actuator><position joint="j" kp="10" kv="1"/></actuator>
    </mujoco>""")
    sim = _step_shell(model, latency=latency)
    sim.data.ctrl[0] = 0.2
    reference = mujoco.MjData(model)
    reference.ctrl[0] = 0.2
    efforts = []
    for _ in range(sim.decimation):
        mujoco.mj_step(model, reference)
        efforts.append(reference.qfrc_actuator[0])
    sim._physics_step()
    assert efforts[-1] != pytest.approx(efforts[0])
    assert efforts[-1] != pytest.approx(sim.data.qfrc_actuator[0])
    torch.testing.assert_close(
        sim._get_simulator_dof_forces().dof_forces,
        torch.tensor([[efforts[-1]]], dtype=torch.float32),
    )


@pytest.mark.parametrize("mode", ["implicit_pd", "explicit_pd", "torque"])
def test_real_backend_effort_survives_snapshot_and_clears_on_reset(mode, monkeypatch):
    """Real control modes report final engine effort and restore it after reset."""
    robot = robot_config("smpl")
    if mode == "torque":
        robot.control.control_type = ControlType.TORQUE
    config = MujocoSimulatorConfig(
        num_envs=1,
        headless=True,
        experiment_name="engine_torque_test",
        sim=robot.simulation_params.mujoco,
        use_implicit_pd=mode != "explicit_pd",
    )
    scene = SceneLib.empty(num_envs=1, device="cpu", terrain=None)
    sim = MujocoSimulator(config, robot, None, torch.device("cpu"), scene)
    try:
        sim._initialize_with_markers({})
        count = sim._num_actuated_dofs
        torch.testing.assert_close(
            sim.get_dof_forces().dof_forces, torch.zeros(1, count)
        )
        command = torch.full((1, count), 0.05 if mode != "torque" else 2.0)
        apply = (
            sim._apply_simulator_torques
            if mode == "torque"
            else sim._apply_simulator_pd_targets
        )
        sim._apply_control = lambda: apply(command)
        engine_efforts = []
        step = mujoco.mj_step

        def record_step(model, data):
            step(model, data)
            engine_efforts.append((data.qfrc_actuator + data.qfrc_applied).copy())

        monkeypatch.setattr(mujoco, "mj_step", record_step)
        sim._physics_step()
        assert len(engine_efforts) == sim.decimation
        expected = torch.tensor(engine_efforts[-1][6 : 6 + count], dtype=torch.float32)[
            None
        ]
        torch.testing.assert_close(sim._get_simulator_dof_forces().dof_forces, expected)
        assert torch.any(expected.abs() > 0.01)
        torch.testing.assert_close(
            sim.get_dof_forces().dof_forces,
            expected[:, sim.data_conversion.dof_convert_to_common],
        )
        sim._apply_torques_to_ctrl(np.zeros(count))
        torch.testing.assert_close(sim._get_simulator_dof_forces().dof_forces, expected)
        owner = SimulatorPhysicalState(sim, scene, 1, torch.device("cpu"))
        snapshot = owner.save_state()
        saved_effort = snapshot["simulator_backend_state"]["applied_dof_forces"]
        sim._applied_dof_forces.fill(999.0)
        torch.testing.assert_close(saved_effort.float()[None], expected)
        reset = ResetState.from_robot_state(snapshot["robot_state"].clone())
        reset.dof_pos.fill_(0.2)
        sim.reset_envs(reset, env_ids=torch.tensor([0]))
        torch.testing.assert_close(
            sim.get_dof_forces().dof_forces, torch.zeros(1, count)
        )
        if mode == "implicit_pd":
            assert np.any(sim.data.qfrc_actuator[6 : 6 + count] != 0)
        owner.load_state(snapshot)
        torch.testing.assert_close(sim._get_simulator_dof_forces().dof_forces, expected)
        saved_effort.fill_(42.0)
        torch.testing.assert_close(sim._get_simulator_dof_forces().dof_forces, expected)
        sim._applied_dof_forces.fill(0.0)
        assert torch.all(saved_effort == 42.0)
    finally:
        sim.shutdown()


@pytest.mark.parametrize("saved", [torch.zeros(2, dtype=torch.float64), torch.zeros(1)])
def test_effort_snapshot_rejects_incompatible_shape_or_dtype(saved):
    """A backend snapshot must match this simulator's effort buffer contract."""
    sim = object.__new__(MujocoSimulator)
    sim._applied_dof_forces = np.zeros(1)
    with pytest.raises(ValueError, match="effort snapshot does not match"):
        sim.restore_backend_state({"applied_dof_forces": saved})
