# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU boundary checks for measured Isaac Lab link/COM velocity references."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from protomotions.simulator.base_simulator.simulator import (
    convert_linear_velocity_reference,
)
from protomotions.simulator.base_simulator.simulator_state import (
    ObjectState,
    ResetState,
    RobotState,
    RootOnlyState,
    StateConversion,
)


def _matrix(q):
    """Use an independent quaternion-to-matrix formula for the oracle."""
    x, y, z, w = q.unbind(-1)
    return torch.stack(
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
    ).reshape(*q.shape[:-1], 3, 3)


def _pose_derivative(q, com_b, com_velocity, omega):
    """Advance world COM and orientation, then differentiate link-origin position."""
    dt = 1e-6
    offset = (_matrix(q) @ com_b.unsqueeze(-1)).squeeze(-1)
    x, y, z = omega.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1)
    skew = skew.reshape(*omega.shape[:-1], 3, 3)
    next_offset = (torch.matrix_exp(skew * dt) @ offset.unsqueeze(-1)).squeeze(-1)
    return (offset + com_velocity * dt - next_offset) / dt


class _Robot:
    """Small indexed articulation boundary with explicit native COM buffers."""

    def __init__(self):
        rng = torch.Generator().manual_seed(812)
        draw = lambda *shape: torch.randn(*shape, generator=rng, dtype=torch.float64)
        q = draw(4, 3, 4)
        q /= q.norm(dim=-1, keepdim=True)
        self.data = SimpleNamespace(
            body_pos_w=draw(4, 3, 3),
            body_quat_w=q,
            body_com_pos_b=draw(4, 3, 3) * 0.2,
            body_lin_vel_w=draw(4, 3, 3),
            body_ang_vel_w=draw(4, 3, 3),
            joint_pos=draw(4, 2),
            joint_vel=draw(4, 2),
        )
        self.writes = []
        self.refresh()

    def refresh(self):
        d = self.data
        offset = (_matrix(d.body_quat_w) @ d.body_com_pos_b.unsqueeze(-1)).squeeze(-1)
        d.body_link_lin_vel_w = d.body_lin_vel_w - torch.cross(
            d.body_ang_vel_w, offset, dim=-1
        )
        d.root_pos_w = d.body_pos_w[:, 0]
        d.root_quat_w = d.body_quat_w[:, 0]
        d.root_link_lin_vel_w = d.body_link_lin_vel_w[:, 0]
        d.root_ang_vel_w = d.body_ang_vel_w[:, 0]
        d.root_com_vel_w = torch.cat((d.body_lin_vel_w[:, 0], d.root_ang_vel_w), dim=-1)
        d.root_vel_w = d.root_com_vel_w

    def write_root_link_pose_to_sim_index(self, root_pose, env_ids):
        selected = slice(None) if env_ids is None else env_ids
        self.data.body_pos_w[selected, 0] = root_pose[:, :3]
        self.data.body_quat_w[selected, 0] = root_pose[:, 3:]
        self.writes.append("pose")
        self.refresh()

    def write_root_com_velocity_to_sim_index(self, root_velocity, env_ids):
        selected = slice(None) if env_ids is None else env_ids
        self.data.body_lin_vel_w[selected, 0] = root_velocity[:, :3]
        self.data.body_ang_vel_w[selected, 0] = root_velocity[:, 3:]
        self.writes.append("com_velocity")
        self.refresh()

    def write_root_velocity_to_sim(self, root_velocity, env_ids):
        # Legacy writer aliases COM semantics; retained for the old-method proof.
        self.write_root_com_velocity_to_sim_index(root_velocity, env_ids)

    def set_joint_position_target(self, positions, joint_ids, env_ids):
        self.writes.append("joint_target")

    def write_joint_state_to_sim(self, positions, velocities, joint_ids, env_ids):
        selected = slice(None) if env_ids is None else env_ids
        self.data.joint_pos[selected] = positions
        self.data.joint_vel[selected] = velocities


@pytest.fixture
def sim():
    """Run real adapter method bodies without importing or launching the optional SDK."""
    path = Path(__file__).parents[1] / "simulator/isaaclab/simulator.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "IsaacLabSimulator"
    )
    names = {
        "_get_simulator_bodies_state",
        "_get_simulator_root_state",
        "_set_simulator_env_state",
        "_apply_root_velocity_impulse",
    }
    methods = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert len(methods) == len(names)
    namespace = dict(
        torch=torch,
        wp=SimpleNamespace(to_torch=lambda value: value),
        Optional=Optional,
        ObjectState=ObjectState,
        ResetState=ResetState,
        RobotState=RobotState,
        RootOnlyState=RootOnlyState,
        StateConversion=StateConversion,
        convert_linear_velocity_reference=convert_linear_velocity_reference,
    )
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace
    )
    adapter = type("AdapterBoundary", (), {name: namespace[name] for name in names})()
    adapter.num_envs, adapter._num_bodies = 4, 3
    adapter._robot = _Robot()
    return adapter


@pytest.mark.parametrize("ids", [None, [3, 1], []])
def test_link_reads_match_pose_derivative_without_second_com_shift(sim, ids):
    """Root and child reads already use link origins and agree with pose derivatives."""
    ids = None if ids is None else torch.tensor(ids, dtype=torch.long)
    selected = slice(None) if ids is None else ids
    d = sim._robot.data
    for change in (0.0, 0.17):
        d.body_com_pos_b += change
        sim._robot.refresh()
        expected = _pose_derivative(
            d.body_quat_w, d.body_com_pos_b, d.body_lin_vel_w, d.body_ang_vel_w
        )
        body = sim._get_simulator_bodies_state(ids)
        root = sim._get_simulator_root_state(ids)
        torch.testing.assert_close(
            body.rigid_body_vel, expected[selected], atol=3e-6, rtol=0
        )
        torch.testing.assert_close(root.root_vel, body.root_vel, atol=0, rtol=0)
        torch.testing.assert_close(
            body.rigid_body_pos, d.body_pos_w[selected], atol=0, rtol=0
        )
        torch.testing.assert_close(
            body.rigid_body_rot, d.body_quat_w[selected], atol=0, rtol=0
        )
        torch.testing.assert_close(
            body.rigid_body_ang_vel, d.body_ang_vel_w[selected], atol=0, rtol=0
        )
        original = d.body_link_lin_vel_w.clone()
        body.rigid_body_vel.add_(1.0)
        root.root_vel.add_(1.0)
        torch.testing.assert_close(d.body_link_lin_vel_w, original, atol=0, rtol=0)


@pytest.mark.parametrize("ids", [None, [3, 1], []])
def test_reset_inverse_uses_new_pose_live_com_and_selected_worlds(sim, ids):
    """Reset converts new link-origin velocity to COM after using the new orientation."""
    selected = torch.arange(4) if ids is None else torch.tensor(ids, dtype=torch.long)
    d = sim._robot.data
    before = [
        value.clone()
        for value in (
            d.body_pos_w,
            d.body_quat_w,
            d.body_lin_vel_w,
            d.body_ang_vel_w,
            d.joint_pos,
            d.joint_vel,
        )
    ]
    d.body_com_pos_b += 0.23
    state = ResetState(
        root_pos=d.root_pos_w[selected] + 2.0,
        root_rot=d.root_quat_w[selected].roll(1, -1),
        root_vel=d.root_link_lin_vel_w[selected] + 0.7,
        root_ang_vel=d.root_ang_vel_w[selected] - 0.4,
        dof_pos=d.joint_pos[selected] + 0.3,
        dof_vel=d.joint_vel[selected] - 0.2,
        state_conversion=StateConversion.SIMULATOR,
    )
    sim._set_simulator_env_state(state, env_ids=None if ids is None else selected)
    derivative = _pose_derivative(
        d.root_quat_w, d.body_com_pos_b[:, 0], d.body_lin_vel_w[:, 0], d.root_ang_vel_w
    )
    torch.testing.assert_close(derivative[selected], state.root_vel, atol=3e-6, rtol=0)
    torch.testing.assert_close(d.root_pos_w[selected], state.root_pos, atol=0, rtol=0)
    torch.testing.assert_close(d.root_quat_w[selected], state.root_rot, atol=0, rtol=0)
    torch.testing.assert_close(
        d.root_ang_vel_w[selected], state.root_ang_vel, atol=0, rtol=0
    )
    torch.testing.assert_close(d.joint_pos[selected], state.dof_pos, atol=0, rtol=0)
    torch.testing.assert_close(d.joint_vel[selected], state.dof_vel, atol=0, rtol=0)
    untouched = torch.tensor(
        [i for i in range(4) if i not in selected], dtype=torch.long
    )
    for old, actual in zip(
        before,
        (
            d.body_pos_w,
            d.body_quat_w,
            d.body_lin_vel_w,
            d.body_ang_vel_w,
            d.joint_pos,
            d.joint_vel,
        ),
    ):
        torch.testing.assert_close(actual[untouched], old[untouched], atol=0, rtol=0)
    assert sim._robot.writes[:2] == ["pose", "com_velocity"]


@pytest.mark.parametrize("ids", [[3, 1], []])
@pytest.mark.parametrize("kind", ["mixed", "angular_only", "zero_com"])
def test_root_impulse_changes_origin_derivative_by_requested_delta(sim, ids, kind):
    """Angular pushes preserve the requested origin translation and untouched worlds."""
    ids = torch.tensor(ids, dtype=torch.long)
    d = sim._robot.data
    if kind == "zero_com":
        d.body_com_pos_b.zero_()
    else:
        d.body_com_pos_b += 0.13
    sim._robot.refresh()
    before = d.root_com_vel_w.clone()
    old_derivative = _pose_derivative(
        d.root_quat_w, d.body_com_pos_b[:, 0], before[:, :3], before[:, 3:]
    )
    dv = torch.tensor([0.4, -0.2, 0.1], dtype=torch.float64).repeat(len(ids), 1)
    if kind == "angular_only":
        dv.zero_()
    dw = torch.tensor([-0.3, 0.5, 0.7], dtype=torch.float64).repeat(len(ids), 1)
    sim._apply_root_velocity_impulse(dv, dw, ids)
    after = d.root_com_vel_w
    derivative = _pose_derivative(
        d.root_quat_w, d.body_com_pos_b[:, 0], after[:, :3], after[:, 3:]
    )
    torch.testing.assert_close(
        derivative[ids], old_derivative[ids] + dv, atol=3e-6, rtol=0
    )
    torch.testing.assert_close(after[ids, 3:], before[ids, 3:] + dw, atol=0, rtol=0)
    untouched = torch.tensor([i for i in range(4) if i not in ids], dtype=torch.long)
    torch.testing.assert_close(after[untouched], before[untouched], atol=0, rtol=0)
    assert sim._robot.writes == ["com_velocity"]
