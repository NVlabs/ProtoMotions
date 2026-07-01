# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from protomotions.simulator.base_simulator.config import SimParams, SimulatorConfig


@dataclass
class NewtonSimParams(SimParams):
    """Newton/MuJoCo solver parameters."""

    solver: str = field(
        default="newton",
        metadata={"help": "Constraint solver: 'newton', 'cg', or 'direct'."}
    )
    integrator: str = field(
        default="implicitfast",
        metadata={"help": "Integrator: 'euler', 'implicit', or 'implicitfast'."}
    )
    iterations: int = field(
        default=100,
        metadata={"help": "Max solver iterations."}
    )
    ls_iterations: int = field(
        default=50,
        metadata={"help": "Line search iterations."}
    )
    ls_parallel: bool = field(
        default=True,
        metadata={"help": "Run line search in parallel."}
    )
    impratio: float = field(
        default=10.0,
        metadata={"help": "Implicit integration ratio."}
    )
    njmax: int = field(
        default=450,
        metadata={"help": "Max constraint Jacobian rows."}
    )
    nconmax: int = field(
        default=300,
        metadata={"help": "Max contacts."}
    )
    cone: str = field(
        default="pyramidal",
        metadata={"help": "Friction cone: 'pyramidal' or 'elliptic'."}
    )
    ccd_iterations: int = field(
        default=200,
        metadata={"help": "CCD (continuous collision detection) iterations."}
    )


@dataclass
class NewtonSimulatorConfig(SimulatorConfig):
    """Configuration specific to Newton simulator."""

    _target_: str = "protomotions.simulator.newton.simulator.NewtonSimulator"
    sim: NewtonSimParams = field(default_factory=NewtonSimParams)  # Override sim type
    w_last: bool = True  # Newton uses xyzw quaternions
