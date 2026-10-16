# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Newton must step in its explicit control modes, TORQUE and PROPORTIONAL.

Actions reach the simulator already processed: PD targets for PROPORTIONAL,
torques for TORQUE. The other backends go through the base ``_apply_control``,
which uses them as-is. Newton has its own path, because it applies these modes
from Warp kernels captured in a CUDA graph: it writes the actions into a
persistent Warp buffer that the kernels read on every substep. That path kept
calling the scaling helpers the ActionProcessor refactor deleted, so the first
step in either mode raised AttributeError. These tests step a real G1 and check
what landed in that buffer.

Runs on CUDA when available, otherwise CPU. On CPU the first run spends about
a minute compiling Warp kernels; later runs take a few seconds each.
"""

import pytest
import torch

from protomotions.robot_configs.base import ControlType

NUM_ENVS = 2


def _make_simulator(control_type: ControlType):
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    from protomotions.components.scene_lib import SceneLib
    from protomotions.components.terrains.config import TerrainConfig
    from protomotions.components.terrains.terrain import Terrain
    from protomotions.robot_configs.factory import robot_config
    from protomotions.simulator.newton.config import (
        NewtonSimParams,
        NewtonSimulatorConfig,
    )
    from protomotions.simulator.newton.simulator import NewtonSimulator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    robot_cfg = robot_config("g1")
    robot_cfg.control.control_type = control_type
    terrain = Terrain(
        config=TerrainConfig(
            map_length=20.0,
            map_width=20.0,
            border_size=40.0,
            num_levels=1,
            num_terrains=1,
            terrain_proportions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            horizontal_scale=0.1,
            vertical_scale=0.005,
        ),
        num_envs=NUM_ENVS,
        device=device,
    )
    simulator = NewtonSimulator(
        config=NewtonSimulatorConfig(
            num_envs=NUM_ENVS,
            headless=True,
            sim=NewtonSimParams(fps=120, decimation=4),
            experiment_name="newton_explicit_control_test",
        ),
        robot_config=robot_cfg,
        terrain=terrain,
        device=device,
        scene_lib=SceneLib.empty(
            num_envs=NUM_ENVS, device=str(device), terrain=terrain
        ),
    )
    simulator._initialize_with_markers({})
    return simulator


def _control_buffer(simulator) -> torch.Tensor:
    import warp as wp

    return wp.to_torch(simulator._pd_targets_wp).view(simulator.num_envs, -1)


@pytest.mark.needs_lfs
def test_torque_mode_applies_actions_as_clipped_torques():
    simulator = _make_simulator(ControlType.TORQUE)
    limits = simulator._torque_limits_common
    # Distinct per-joint values so a wrong DOF reordering shows up, and the
    # outer joints ask for up to twice their limit so clipping is exercised.
    ramp = torch.linspace(-2.0, 2.0, limits.numel(), device=simulator.device)
    actions = (ramp * limits).expand(NUM_ENVS, -1).clone()

    simulator.step(actions)

    expected = torch.clip(actions, -limits, limits)[
        :, simulator.data_conversion.dof_convert_to_sim
    ]
    torch.testing.assert_close(_control_buffer(simulator), expected)


@pytest.mark.needs_lfs
def test_proportional_mode_applies_actions_as_pd_targets():
    simulator = _make_simulator(ControlType.PROPORTIONAL)
    num_dofs = simulator.robot_config.number_of_actions
    targets = torch.linspace(-0.3, 0.3, num_dofs, device=simulator.device)
    actions = targets.expand(NUM_ENVS, -1).clone()

    simulator.step(actions)

    expected = actions[:, simulator.data_conversion.dof_convert_to_sim]
    torch.testing.assert_close(_control_buffer(simulator), expected)
