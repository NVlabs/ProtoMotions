# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gym backend boundary contracts without importing proprietary GPU bindings."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace, MethodType
from typing import Optional

import torch

from protomotions.simulator.base_simulator.simulator_state import (
    DataConversionMapping,
    ObjectState,
    ResetState,
    RobotState,
    RootOnlyState,
    StateConversion,
)
from protomotions.tests.support.sensor_simulator import SensorTestSimulator


def backend_methods(fake):
    path = Path(__file__).parents[1] / "simulator/isaacgym/simulator.py"
    cls = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)
    )
    names = (
        "_set_simulator_env_state",
        "_refresh_reset_body_state",
        "_flush_pending_state",
        "_get_simulator_bodies_state",
        "_get_simulator_root_state",
    )
    namespace = dict(
        torch=torch,
        Optional=Optional,
        RobotState=RobotState,
        ResetState=ResetState,
        RootOnlyState=RootOnlyState,
        ObjectState=ObjectState,
        StateConversion=StateConversion,
        gymtorch=SimpleNamespace(unwrap_tensor=lambda x: x),
    )
    methods = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace
    )
    for name in names:
        setattr(fake, name, MethodType(namespace[name], fake))


class LastSetterWinsGym:
    """Unlike a generic mock, two setters before simulate DROP the first ids."""

    def __init__(self):
        self.calls = {"root": [], "dof": []}
        self.pending = {}

    def set_actor_root_state_tensor_indexed(self, sim, values, ids, count):
        self.calls["root"].append(ids.clone())
        self.pending["root"] = (values, ids.clone())

    def set_dof_state_tensor_indexed(self, sim, values, ids, count):
        self.calls["dof"].append(ids.clone())
        self.pending["dof"] = (values, ids.clone())


def make_backend():
    oracle = SensorTestSimulator(3)
    # Deliberately reorder BOTH bodies and joints at the backend boundary.
    bodies, dofs = torch.tensor([0, 3, 1, 2]), torch.tensor([1, 0])
    mapping = DataConversionMapping(
        body_convert_to_common=bodies.argsort(),
        body_convert_to_sim=bodies,
        dof_convert_to_common=dofs.argsort(),
        dof_convert_to_sim=dofs,
        sim_w_last=True,
    )
    roots, joints = torch.zeros(3, 13), torch.zeros(3, 2, 2)
    fake = SimpleNamespace(
        num_envs=3,
        device=torch.device("cpu"),
        _sim=None,
        _gym=LastSetterWinsGym(),
        _root_states=roots,
        _humanoid_root_states=roots,
        _humanoid_actor_ids=torch.arange(3, dtype=torch.int32),
        _dof_state=joints,
        _dof_pos=joints[:, :, 0],
        _dof_vel=joints[:, :, 1],
        _rigid_body_pos=torch.full((3, 4, 3), -100.0),
        _rigid_body_rot=torch.zeros(3, 4, 4),
        _rigid_body_vel=torch.zeros(3, 4, 3),
        _rigid_body_ang_vel=torch.zeros(3, 4, 3),
        _body_com_offsets=torch.tensor([0.03, -0.02, -0.076]).expand(3, 4, 3).clone(),
        _pending_root_ids=[],
        _pending_dof_ids=[],
        data_conversion=mapping,
        robot_config=oracle.robot_config,
    )
    backend_methods(fake)
    return oracle, fake


def test_reset_is_fresh_before_simulate_and_preserves_unselected_rows():
    oracle, backend = make_backend()
    ids = torch.tensor([2, 0])
    expected = oracle.state[ids].clone()
    reset = ResetState.from_robot_state(expected).convert_to_sim(
        backend.data_conversion
    )
    backend._set_simulator_env_state(reset, env_ids=ids)
    actual = backend._get_simulator_bodies_state(ids).convert_to_common(
        backend.data_conversion
    )
    for name in (
        "rigid_body_pos",
        "rigid_body_rot",
        "rigid_body_vel",
        "rigid_body_ang_vel",
    ):
        torch.testing.assert_close(
            getattr(actual, name), getattr(expected, name), atol=1e-6, rtol=1e-6
        )
    torch.testing.assert_close(
        backend._get_simulator_root_state(ids).root_vel, expected.root_vel
    )
    assert (backend._rigid_body_pos[1] == -100.0).all()
    assert not hasattr(
        backend, "_contact_forces"
    )  # Gym contact reporting is unsupported.
    assert not backend._gym.calls["root"]  # No eager setter/physics step.


def test_restore_then_partial_reset_flushes_union_once_with_latest_values():
    oracle, backend = make_backend()
    snapshot = ResetState.from_robot_state(oracle.state.clone())
    backend._set_simulator_env_state(
        snapshot.convert_to_sim(backend.data_conversion), env_ids=torch.arange(3)
    )
    root_before = backend._root_states.clone()
    reset = ResetState.from_robot_state(oracle.state[[1]].clone())
    reset.root_pos += 50.0
    backend._set_simulator_env_state(
        reset.convert_to_sim(backend.data_conversion), env_ids=torch.tensor([1])
    )
    backend._flush_pending_state()
    for kind in ("root", "dof"):
        assert len(backend._gym.calls[kind]) == 1
        assert backend._gym.calls[kind][0].tolist() == [0, 1, 2]
    values, ids = backend._gym.pending["root"]
    restored = torch.full_like(values, -999.0)
    restored[ids] = values[ids]
    torch.testing.assert_close(restored[[0, 2]], root_before[[0, 2]])
    torch.testing.assert_close(restored[1, :3], root_before[1, :3] + 50.0)
    backend._flush_pending_state()
    assert len(backend._gym.calls["root"]) == 1


def test_coupled_joint_matches_native_gym_probe_not_euler_or_parent_frame():
    from protomotions.components.pose_lib import extract_kinematic_info
    from protomotions.simulator.isaacgym.reset_kinematics import reset_body_state

    ki = extract_kinematic_info(
        str(Path(__file__).parent / "fixtures/gym_spherical_chain.xml")
    )
    state = ResetState(
        state_conversion=StateConversion.COMMON,
        root_pos=torch.tensor([[0.0, 0.0, 2.0]]),
        root_rot=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        root_vel=torch.zeros(1, 3),
        root_ang_vel=torch.zeros(1, 3),
        dof_pos=torch.tensor([[0.3, -0.4, 0.2]]),
        dof_vel=torch.tensor([[0.7, -0.2, 0.5]]),
    )
    _, quat, _, omega = reset_body_state(ki, state)
    # Captured from a native Gym CPU-PhysX one-step probe (dt=1e-6, no gravity).
    # The nonidentity rest rotation distinguishes parent and child conventions.
    torch.testing.assert_close(
        quat[0, 1],
        torch.tensor([0.19428506, -0.15250386, 0.34492302, 0.90555173]),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        omega[0, 1],
        torch.tensor([0.56658185, 0.02995334, 0.67682171]),
        atol=1e-6,
        rtol=0,
    )


def test_read_path_turns_native_gym_com_velocity_into_link_origin_velocity():
    """Gym reports centre-of-mass velocity; the backend must report link origins.

    The fixture holds raw tensors captured from native GPU Gym one physics step
    apart: G1 released in the air with a spinning root, physics at 1000 Hz,
    tensors read straight from Gym without the backend's getters. Link-origin
    velocity is the central difference of link-origin positions, so this
    checks the sign of the conversion against the real engine, not against
    the backend's own write path.
    """
    probe = json.loads(
        (Path(__file__).parent / "fixtures/gym_com_velocity_probe.json").read_text()
    )
    _, backend = make_backend()

    def tensor(key):
        return torch.tensor(probe[key])[None]

    origin_vel = (tensor("pos_after") - tensor("pos_before")) / (2 * probe["dt"])
    backend._rigid_body_pos = (tensor("pos_before") + tensor("pos_after")) / 2
    backend._rigid_body_rot = tensor("rot_xyzw")
    backend._rigid_body_vel = tensor("raw_lin_vel")
    backend._rigid_body_ang_vel = tensor("ang_vel")
    backend._body_com_offsets = tensor("com_offset_body")
    # Gym's root-state tensor and body 0 of the rigid-body tensor hold the same
    # raw velocity (checked on the GPU when the fixture was captured).
    backend._humanoid_root_states = torch.cat(
        [
            backend._rigid_body_pos[:, 0],
            backend._rigid_body_rot[:, 0],
            backend._rigid_body_vel[:, 0],
            backend._rigid_body_ang_vel[:, 0],
        ],
        dim=-1,
    )

    raw_error = (backend._rigid_body_vel - origin_vel).norm(dim=-1)
    assert raw_error.min() > 0.2  # The fixture can tell the two conventions apart.
    bodies = backend._get_simulator_bodies_state()
    torch.testing.assert_close(bodies.rigid_body_vel, origin_vel, atol=0.01, rtol=0)
    root = backend._get_simulator_root_state()
    torch.testing.assert_close(root.root_vel, origin_vel[:, 0], atol=0.01, rtol=0)
