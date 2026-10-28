# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Optional
from protomotions.simulator.base_simulator.config import SimParams, SimulatorConfig


@dataclass
class MujocoSimParams(SimParams):
    """MuJoCo simulation parameters.

    Defaults match RoboJuDo: 1000 Hz physics; at the humanoid control_hz=50 that
    is 20 physics steps per control step (derived in Simulator.__init__). No
    solver params -- we use MuJoCo's built-in defaults for everything (solver,
    iterations, etc.) since RoboJuDo does the same. The integrator can be
    overridden per robot.
    """

    fps: Optional[int] = field(
        default=None,
        metadata={
            "help": "Simulation frames per second (physics timestep = 1/fps). "
            "None derives it as control_hz * 20 (1000 Hz at the humanoid 50 Hz)."
        },
    )
    integrator: Optional[str] = field(
        default=None,
        metadata={
            "help": "MuJoCo integrator: 'euler', 'implicitfast', 'implicit' or "
            "'rk4'. None keeps the MJCF's choice (MuJoCo's default is Euler). "
            "Euler applies actuator damping explicitly, which can go unstable "
            "with stiff PD gains on light links; 'implicitfast' does not."
        },
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
        },
    )
