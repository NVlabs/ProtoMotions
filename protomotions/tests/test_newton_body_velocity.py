# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contract tests for Newton's COM/world to common link-origin adapter."""

import importlib
import importlib.metadata
import math
from types import SimpleNamespace

import pytest
import torch

from protomotions.simulator.base_simulator.simulator import (
    convert_linear_velocity_reference,
)
from protomotions.simulator.base_simulator.simulator_state import ResetState


@pytest.fixture
def backend():
    """Load the optional Newton runtime only for backend contract tests."""
    pytest.importorskip("newton")
    return importlib.import_module("protomotions.simulator.newton.simulator")


def _offsets(rot: torch.Tensor, com: torch.Tensor) -> torch.Tensor:
    """Independent quaternion-to-matrix oracle (no production rotation helper)."""
    x, y, z, w = rot.unbind(-1)
    matrix = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*rot.shape[:-1], 3, 3)
    return (matrix @ com.unsqueeze(-1)).squeeze(-1)


class _View:
    """ArticulationView-shaped arrays, with masked writes and copied reads."""

    def get_attribute(self, name: str, model: SimpleNamespace) -> torch.Tensor:
        assert name == "body_com"
        return model.com.clone().unsqueeze(1)

    def get_link_transforms(self, state: SimpleNamespace) -> torch.Tensor:
        return state.pose.clone().unsqueeze(1)

    def get_link_velocities(self, state: SimpleNamespace) -> torch.Tensor:
        return state.twist.clone().unsqueeze(1)

    def get_root_transforms(self, state: SimpleNamespace) -> torch.Tensor:
        return state.pose[:, :1].clone()

    def get_root_velocities(self, state: SimpleNamespace) -> torch.Tensor:
        return state.twist[:, :1].clone()

    def get_dof_positions(self, state: SimpleNamespace) -> torch.Tensor:
        return state.dof_pos.clone().unsqueeze(1)

    def get_dof_velocities(self, state: SimpleNamespace) -> torch.Tensor:
        return state.dof_vel.clone().unsqueeze(1)

    def set_root_transforms(
        self, state: SimpleNamespace, value: torch.Tensor, mask: torch.Tensor
    ) -> None:
        state.pose[mask, 0] = value[mask, 0]

    def set_root_velocities(
        self, state: SimpleNamespace, value: torch.Tensor, mask: torch.Tensor
    ) -> None:
        state.twist[mask, 0] = value[mask, 0]

    def set_dof_positions(
        self, state: SimpleNamespace, value: torch.Tensor, mask: torch.Tensor
    ) -> None:
        state.dof_pos[mask] = value[mask, 0]

    def set_dof_velocities(
        self, state: SimpleNamespace, value: torch.Tensor, mask: torch.Tensor
    ) -> None:
        state.dof_vel[mask] = value[mask, 0]


@pytest.fixture
def sim(monkeypatch, backend):
    """Exercise production adapter methods with isolated, masked state buffers."""
    rng = torch.Generator().manual_seed(741)
    sim = object.__new__(backend.NewtonSimulator)
    sim.num_envs = 4
    sim.device = torch.device("cpu")
    sim.robot_config = SimpleNamespace(
        kinematic_info=SimpleNamespace(num_bodies=3),
        asset=SimpleNamespace(fix_base_link=False),
    )
    sim.robot_view = _View()
    sim.model = SimpleNamespace(com=torch.randn(4, 3, 3, generator=rng) * 0.2)
    pose = torch.randn(4, 3, 7, generator=rng)
    pose[..., 3:] /= pose[..., 3:].norm(dim=-1, keepdim=True)
    twist = torch.randn(4, 3, 6, generator=rng)

    def state() -> SimpleNamespace:
        result = SimpleNamespace(
            pose=pose.clone(),
            twist=twist.clone(),
            dof_pos=torch.randn(4, 2, generator=rng),
            dof_vel=torch.randn(4, 2, generator=rng),
            joint_q=None,
            joint_qd=None,
            clears=0,
        )

        def clear() -> None:
            result.clears += 1

        result.clear_forces = clear
        return result

    sim.state_0, sim.state_1 = state(), state()
    sim.solver = SimpleNamespace(
        mjw_data=SimpleNamespace(qacc_warmstart=torch.ones(4, 2))
    )
    sim.fk_calls = []
    monkeypatch.setattr(backend.wp, "to_torch", lambda value: value)
    monkeypatch.setattr(
        backend.newton, "eval_fk", lambda *args: sim.fk_calls.append(args[-1])
    )
    return sim


@pytest.mark.parametrize("ids", [None, [3, 1], []])
@pytest.mark.parametrize("case", ["nonzero", "zero_com", "zero_omega"])
def test_reads_match_origin_pose_derivative_and_root(sim, ids, case) -> None:
    """Convert root reads with live COM offsets while preserving every child value."""
    if case == "zero_com":
        sim.model.com.zero_()
    if case == "zero_omega":
        sim.state_0.twist[..., 3:].zero_()
    ids = None if ids is None else torch.tensor(ids, dtype=torch.long)
    for change in (0.0, 0.13):
        # A second read must observe changed, per-environment randomized COMs.
        sim.model.com.add_(change)
        before = sim.state_0.twist.clone()
        offset = _offsets(sim.state_0.pose[..., 3:], sim.model.com)
        omega = before[..., 3:]
        expected = before[..., :3].clone()
        expected[:, 0] -= torch.cross(omega[:, 0], offset[:, 0], dim=-1)
        body = sim._get_simulator_bodies_state(ids)
        root = sim._get_simulator_root_state(ids)
        selected = slice(None) if ids is None else ids
        torch.testing.assert_close(body.rigid_body_vel, expected[selected])
        torch.testing.assert_close(root.root_vel, body.root_vel)
        torch.testing.assert_close(root.root_ang_vel, omega[selected, 0])
        torch.testing.assert_close(
            body.rigid_body_pos, sim.state_0.pose[selected, :, :3]
        )
        torch.testing.assert_close(
            body.rigid_body_rot, sim.state_0.pose[selected, :, 3:]
        )
        torch.testing.assert_close(sim.state_0.twist, before)


@pytest.mark.parametrize("ids", [None, [3, 1], []])
def test_reset_inverse_uses_new_rotation_and_live_com_in_both_buffers(sim, ids) -> None:
    """Reset uses the new orientation and live COM without touching other worlds."""
    selected = torch.arange(4) if ids is None else torch.tensor(ids, dtype=torch.long)
    initial = sim._get_simulator_root_state(selected)
    new = ResetState(
        state_conversion=initial.state_conversion,
        root_pos=initial.root_pos + 2,
        root_rot=initial.root_rot.roll(1, dims=-1),
        root_vel=initial.root_vel + 0.7,
        root_ang_vel=initial.root_ang_vel - 0.4,
        dof_pos=torch.full((len(selected), 2), 0.3),
        dof_vel=torch.full((len(selected), 2), -0.5),
    )
    sim.model.com.add_(0.17)
    untouched = torch.tensor(
        [i for i in range(4) if i not in selected], dtype=torch.long
    )
    snapshots = [
        (s.pose.clone(), s.twist.clone(), s.dof_pos.clone(), s.dof_vel.clone())
        for s in (sim.state_0, sim.state_1)
    ]
    sim._set_simulator_env_state(new, env_ids=None if ids is None else selected)
    com_velocity = new.root_vel + torch.cross(
        new.root_ang_vel, _offsets(new.root_rot, sim.model.com[selected, 0]), dim=-1
    )
    for state, snapshot in zip((sim.state_0, sim.state_1), snapshots):
        for actual, old in zip(
            (state.pose, state.twist, state.dof_pos, state.dof_vel), snapshot
        ):
            torch.testing.assert_close(
                actual[untouched], old[untouched], rtol=0, atol=0
            )
        torch.testing.assert_close(state.pose[selected, 0, :3], new.root_pos)
        torch.testing.assert_close(state.pose[selected, 0, 3:], new.root_rot)
        torch.testing.assert_close(state.twist[selected, 0, :3], com_velocity)
        torch.testing.assert_close(state.twist[selected, 0, 3:], new.root_ang_vel)
        torch.testing.assert_close(state.dof_pos[selected], new.dof_pos)
        torch.testing.assert_close(state.dof_vel[selected], new.dof_vel)
        assert state.clears == 1
    assert sim.fk_calls[0] is sim.state_0
    assert sim.fk_calls[1] is sim.state_1
    torch.testing.assert_close(
        sim._get_simulator_root_state(selected).root_vel, new.root_vel
    )
    assert torch.all(sim.solver.mjw_data.qacc_warmstart[selected] == 0)
    assert torch.all(sim.solver.mjw_data.qacc_warmstart[untouched] == 1)


@pytest.mark.parametrize("linear", [False, True])
def test_push_adds_common_origin_delta_with_existing_angular_velocity(
    sim, linear: bool
) -> None:
    """Angular pushes preserve the requested link-origin translation in both buffers."""
    ids = torch.tensor([3, 1])
    before = sim._get_simulator_root_state()
    snapshots = [s.twist.clone() for s in (sim.state_0, sim.state_1)]
    delta_v = torch.full((2, 3), 0.7 if linear else 0.0)
    delta_w = torch.tensor([[0.3, -0.8, 1.1], [-0.5, 0.9, -0.2]])
    sim._apply_root_velocity_impulse(delta_v, delta_w, ids)
    after = sim._get_simulator_root_state()
    torch.testing.assert_close(after.root_vel[ids], before.root_vel[ids] + delta_v)
    torch.testing.assert_close(
        after.root_ang_vel[ids], before.root_ang_vel[ids] + delta_w
    )
    torch.testing.assert_close(after.root_pos, before.root_pos, rtol=0, atol=0)
    torch.testing.assert_close(after.root_rot, before.root_rot, rtol=0, atol=0)
    for state, snapshot in zip((sim.state_0, sim.state_1), snapshots):
        torch.testing.assert_close(
            state.twist[[0, 2]], snapshot[[0, 2]], rtol=0, atol=0
        )
        torch.testing.assert_close(state.twist[ids, 0], sim.state_0.twist[ids, 0])


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 4, 3), (0, 3)])
def test_declared_reference_shift_is_reversible(shape):
    """Reference conversion follows rigid-body kinematics for each declared point."""
    rng = torch.Generator().manual_seed(453)
    velocity, omega, source, target = [
        torch.randn(shape, generator=rng) for _ in range(4)
    ]
    result = convert_linear_velocity_reference(velocity, omega, source, target)
    expected = (
        velocity
        + torch.cross(omega, target, dim=-1)
        - torch.cross(omega, source, dim=-1)
    )
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(
        convert_linear_velocity_reference(result, omega, target, source), velocity
    )
    torch.testing.assert_close(
        convert_linear_velocity_reference(velocity, omega, source, source), velocity
    )


def test_fixed_root_read_declares_link_origin(sim):
    """A fixed root has no COM reference shift and leaves child velocities unchanged."""
    sim.robot_config.asset.fix_base_link = True
    torch.testing.assert_close(
        sim._get_simulator_bodies_state().rigid_body_vel, sim.state_0.twist[..., :3]
    )
    torch.testing.assert_close(
        sim._get_simulator_root_state().root_vel, sim.state_0.twist[:, 0, :3]
    )


_TWO_LINK_XML = """<mujoco model="velocity_probe">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="base" pos="0 0 2">
      <freejoint name="root"/>
      <inertial pos="0.2 0.1 0.05" mass="2" diaginertia="0.3 0.4 0.5"/>
      <geom type="box" size="0.1 0.1 0.1" contype="0" conaffinity="0"/>
      <body name="child" pos="0.5 0.2 0.1">
        <joint name="hinge" type="hinge" axis="0 0 1"/>
        <inertial pos="0.15 -0.1 0.05" mass="1" diaginertia="0.1 0.2 0.2"/>
        <geom type="box" size="0.1 0.1 0.1" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""


@pytest.mark.parametrize(
    "use_mujoco_cpu", [True, False], ids=["mujoco-c", "mujoco-warp"]
)
def test_real_pose_derivative_and_pinned_per_body_reference(
    backend, tmp_path, use_mujoco_cpu
):
    """Both CPU solver routes report COM root velocity and link-origin child velocity."""
    import numpy as np
    from newton.selection import ArticulationView

    assert (
        importlib.metadata.version("newton") == "1.0.0"
    ), "Re-measure and declare each body's velocity reference before upgrading Newton"
    wp, newton = backend.wp, backend.newton
    xml = tmp_path / "two_link.xml"
    xml.write_text(_TWO_LINK_XML)
    with wp.ScopedDevice("cpu"):
        builder = newton.ModelBuilder()
        builder.add_mjcf(str(xml), collapse_fixed_joints=False, floating=True)
        builder.articulation_label = ["robot"]
        model = builder.finalize(device="cpu")
        model.set_gravity((0, 0, 0))
        view = ArticulationView(
            model,
            pattern="robot",
            include_links=["base", "child"],
            include_joints=["hinge"],
        )
        solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=use_mujoco_cpu)
        sim = object.__new__(backend.NewtonSimulator)
        sim.model, sim.robot_view = model, view
        sim.num_envs = 1
        sim.robot_config = SimpleNamespace(
            kinematic_info=SimpleNamespace(num_bodies=2),
            asset=SimpleNamespace(fix_base_link=False),
        )
        sim.state_0, sim.state_1 = model.state(), model.state()
        sim.state_0.joint_q.assign(
            np.array(
                [0, 0, 2, 0, 0, math.sin(0.2), math.cos(0.2), 0.25], dtype=np.float32
            )
        )
        sim.state_0.joint_qd.assign(
            np.array([0.2, -0.1, 0.3, 0.4, -0.5, 0.8, 0.7], dtype=np.float32)
        )
        newton.eval_fk(model, sim.state_0.joint_q, sim.state_0.joint_qd, sim.state_0)
        dt = 0.001
        for phase in ("after_fk", "after_solver_1", "after_solver_2"):
            before = sim._get_simulator_bodies_state()
            positions = before.rigid_body_pos.clone()
            raw = wp.to_torch(view.get_link_velocities(sim.state_0)).squeeze(1).clone()
            com = wp.to_torch(view.get_attribute("body_com", model)).squeeze(1)
            shift = torch.cross(
                raw[..., 3:], _offsets(before.rigid_body_rot, com), dim=-1
            )
            sim.state_0.clear_forces()
            solver.step(sim.state_0, sim.state_1, model.control(), model.contacts(), dt)
            next_positions = wp.to_torch(view.get_link_transforms(sim.state_1)).squeeze(
                1
            )[..., :3]
            derivative = (next_positions - positions) / dt
            raw_error = (raw[..., :3] - derivative).norm(dim=-1)
            shifted_error = (raw[..., :3] - shift - derivative).norm(dim=-1)
            # These assertions establish the native convention independently of
            # the adapter, so a pin change cannot silently double-convert children.
            assert raw_error[0, 0] > 0.20, (phase, raw_error)
            assert shifted_error[0, 0] < 0.002, (phase, shifted_error)
            assert raw_error[0, 1] < 0.002, (phase, raw_error)
            assert shifted_error[0, 1] > 0.20, (phase, shifted_error)
            torch.testing.assert_close(
                before.rigid_body_vel, derivative, atol=0.002, rtol=0
            )
            torch.testing.assert_close(
                before.rigid_body_vel[:, 1], raw[:, 1, :3], atol=0, rtol=0
            )
            torch.testing.assert_close(
                sim._get_simulator_root_state().root_vel, before.root_vel
            )
            sim.state_0, sim.state_1 = sim.state_1, sim.state_0
