# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from protomotions.simulator.base_simulator.config import SimParams, SimulatorConfig


@dataclass
class MujocoSimParams(SimParams):
    """MuJoCo simulation parameters.

    Defaults match RoboJuDo: 1000Hz physics, 20x decimation (50Hz control).
    No solver params -- we use MuJoCo's built-in defaults for everything
    (solver, integrator, iterations, etc.) since RoboJuDo does the same.
    """

    fps: int = field(
        default=1000,
        metadata={"help": "Simulation frames per second (physics timestep = 1/fps)."}
    )
    decimation: int = field(
        default=20,
        metadata={"help": "Number of physics steps per control step."}
    )


@dataclass
class MujocoSimulatorConfig(SimulatorConfig):
    """Configuration specific to MuJoCo simulator."""

    _target_: str = "protomotions.simulator.mujoco.simulator.MujocoSimulator"
    sim: MujocoSimParams = field(default_factory=MujocoSimParams)
    w_last: bool = False  # MuJoCo uses wxyz quaternions
    use_implicit_pd: bool = field(
        default=True,
        metadata={
            "help": "PD control mode for BUILT_IN_PD. "
                    "True: MuJoCo position actuators (implicit PD, stable). "
                    "False: explicit PD torque computation at each physics substep."
        }
    )
