# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Four-body hinge chain whose poses and velocities MuJoCo computes independently.

The chain is deliberately not a straight line: the torso hinges about y above
the pelvis, the hand hinges about x off the torso, and the foot hinges about y
below the pelvis. A mounted (non-root) IMU and a displaced localization point
therefore exercise every rotation in the reconstruction.
"""

from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch

from protomotions.components.pose_lib import KinematicInfo, extract_kinematic_info
from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    StateConversion,
)

CHAIN_XML = """
<mujoco model="sensor_chain">
  <compiler angle="radian"/>
  <worldbody>
    <body name="pelvis" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="sphere" size="0.05" mass="5"/>
      <body name="torso" pos="0 0 0.3">
        <joint name="waist" type="hinge" axis="0 1 0"/>
        <geom type="sphere" size="0.04" mass="3"/>
        <body name="hand" pos="0 0.2 0.1">
          <joint name="shoulder" type="hinge" axis="1 0 0"/>
          <geom type="sphere" size="0.03" mass="1"/>
        </body>
      </body>
      <body name="foot" pos="0.05 0 -0.4">
        <joint name="hip" type="hinge" axis="0 1 0"/>
        <geom type="sphere" size="0.03" mass="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

BODY_NAMES = ("pelvis", "torso", "hand", "foot")


@dataclass
class ChainRobot:
    """The attributes of RobotConfig that the sensor configuration reads."""

    kinematic_info: KinematicInfo
    imu_body_name: str = "pelvis"
    imu_mount_quat: tuple = (0.0, 0.0, 0.0, 1.0)
    anchor_body_index: int = 1

    @property
    def resolved_anchor_body_name(self) -> str:
        return self.kinematic_info.body_names[self.anchor_body_index]


def load_chain(tmp_path: Path) -> tuple[mujoco.MjModel, KinematicInfo]:
    path = tmp_path / "sensor_chain.xml"
    path.write_text(CHAIN_XML)
    return mujoco.MjModel.from_xml_path(str(path)), extract_kinematic_info(str(path))


def chain_state(
    model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray
) -> RobotState:
    """Common-format state of one batch of chains from MuJoCo's own kinematics.

    qpos rows are [root xyz, root quat wxyz, waist, shoulder, hip];
    qvel rows are [root lin xyz, root ang xyz (body frame), joint rates].
    Velocities are of each body-frame origin, in world axes.
    """
    data = mujoco.MjData(model)
    body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in BODY_NAMES
    ]
    pos, rot, vel, ang = [], [], [], []
    for q, v in zip(qpos, qvel):
        data.qpos[:] = q
        data.qvel[:] = v
        mujoco.mj_forward(model, data)
        twist = np.zeros(6)
        p, r, lin, w = [], [], [], []
        for body in body_ids:
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, body, twist, 0
            )
            p.append(data.xpos[body].copy())
            r.append(data.xquat[body][[1, 2, 3, 0]].copy())  # wxyz -> xyzw
            w.append(twist[:3].copy())
            lin.append(twist[3:].copy())
        pos.append(p), rot.append(r), vel.append(lin), ang.append(w)
    f = lambda x: torch.tensor(np.array(x), dtype=torch.float32)
    return RobotState(
        state_conversion=StateConversion.COMMON,
        dof_pos=f(qpos[:, 7:]),
        dof_vel=f(qvel[:, 6:]),
        rigid_body_pos=f(pos),
        rigid_body_rot=f(rot),
        rigid_body_vel=f(vel),
        rigid_body_ang_vel=f(ang),
    )


def random_chain_batch(
    rng: np.random.Generator, n: int
) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.zeros((n, 10))
    qpos[:, :3] = rng.uniform(-1, 1, (n, 3)) + np.array([0, 0, 1.0])
    quat = rng.normal(size=(n, 4))
    qpos[:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    qpos[:, 7:] = rng.uniform(-1.2, 1.2, (n, 3))
    qvel = rng.uniform(-2, 2, (n, 9))
    return qpos, qvel
