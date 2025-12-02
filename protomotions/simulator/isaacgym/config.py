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
from dataclasses import dataclass, field
from protomotions.utils.config_builder import ConfigBuilder
from protomotions.simulator.base_simulator.config import SimParams, SimulatorConfig


@dataclass
class IsaacGymPhysXParams(ConfigBuilder):
    """PhysX physics engine parameters."""

    num_threads: int = 4
    solver_type: int = 1  # 0: pgs, 1: tgs
    num_position_iterations: int = 4
    num_velocity_iterations: int = 0
    contact_offset: float = 0.02
    rest_offset: float = 0.0
    bounce_threshold_velocity: float = 0.2
    max_depenetration_velocity: float = 10.0
    default_buffer_size_multiplier: float = 10.0


@dataclass
class IsaacGymFlexParams(ConfigBuilder):
    """Flex physics parameters."""

    num_inner_iterations: int = 10
    warm_start: float = 0.25


@dataclass
class IsaacGymSimParams(SimParams):
    """PhysX-specific simulation parameters used by IsaacGym and IsaacLab."""

    substeps: int
    physx: IsaacGymPhysXParams = field(default_factory=IsaacGymPhysXParams)
    flex: IsaacGymFlexParams = field(default_factory=IsaacGymFlexParams)


@dataclass
class IsaacGymSimulatorConfig(SimulatorConfig):
    """Configuration specific to IsaacGym simulator."""

    _target_: str = "protomotions.simulator.isaacgym.simulator.IsaacGymSimulator"
    w_last: bool = True
    sim: IsaacGymSimParams = field(default_factory=IsaacGymSimParams)
