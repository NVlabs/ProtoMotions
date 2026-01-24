# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Configuration classes for IsaacGym simulator."""

from dataclasses import dataclass, field
from protomotions.simulator.base_simulator.config import SimParams, SimulatorConfig


@dataclass
class IsaacGymPhysXParams:
    """PhysX physics engine parameters."""

    num_threads: int = field(
        default=4,
        metadata={"help": "Number of CPU threads for physics.", "min": 1}
    )
    solver_type: int = field(
        default=1,
        metadata={"help": "Solver type: 0=PGS (Projected Gauss-Seidel), 1=TGS (Temporal Gauss-Seidel)."}
    )
    num_position_iterations: int = field(
        default=4,
        metadata={"help": "Position solver iterations.", "min": 1}
    )
    num_velocity_iterations: int = field(
        default=0,
        metadata={"help": "Velocity solver iterations.", "min": 0}
    )
    contact_offset: float = field(
        default=0.02,
        metadata={"help": "Contact detection offset.", "min": 0.0}
    )
    rest_offset: float = field(
        default=0.0,
        metadata={"help": "Rest offset for contacts."}
    )
    bounce_threshold_velocity: float = field(
        default=0.2,
        metadata={"help": "Velocity threshold for bounce.", "min": 0.0}
    )
    max_depenetration_velocity: float = field(
        default=10.0,
        metadata={"help": "Maximum velocity for depenetration.", "min": 0.0}
    )
    default_buffer_size_multiplier: float = field(
        default=10.0,
        metadata={"help": "GPU buffer size multiplier.", "min": 1.0}
    )


@dataclass
class IsaacGymFlexParams:
    """Flex physics parameters (soft body simulation)."""

    num_inner_iterations: int = field(
        default=10,
        metadata={"help": "Number of inner solver iterations.", "min": 1}
    )
    warm_start: float = field(
        default=0.25,
        metadata={"help": "Warm start coefficient.", "min": 0.0, "max": 1.0}
    )


@dataclass
class IsaacGymSimParams(SimParams):
    """PhysX-specific simulation parameters used by IsaacGym and IsaacLab."""

    substeps: int = field(
        default=2,
        metadata={"help": "Physics substeps per simulation step.", "min": 1}
    )
    physx: IsaacGymPhysXParams = field(
        default_factory=IsaacGymPhysXParams,
        metadata={"help": "PhysX engine parameters."}
    )
    flex: IsaacGymFlexParams = field(
        default_factory=IsaacGymFlexParams,
        metadata={"help": "Flex soft body parameters."}
    )


@dataclass
class IsaacGymSimulatorConfig(SimulatorConfig):
    """Configuration specific to IsaacGym simulator."""

    _target_: str = "protomotions.simulator.isaacgym.simulator.IsaacGymSimulator"
    w_last: bool = field(
        default=True,
        metadata={"help": "Quaternion format: True for xyzw (IsaacGym convention)."}
    )
    sim: IsaacGymSimParams = field(
        default_factory=IsaacGymSimParams,
        metadata={"help": "IsaacGym-specific simulation parameters."}
    )
