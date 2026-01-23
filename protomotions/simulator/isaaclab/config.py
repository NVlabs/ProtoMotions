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
"""Configuration classes for IsaacLab simulator."""

from dataclasses import dataclass, field
from typing import Any
from protomotions.simulator.base_simulator.config import SimParams, SimulatorConfig
from protomotions.simulator.isaacgym.config import IsaacGymPhysXParams
import torch


@dataclass
class ProtoMotionsIsaacLabMarkers:
    """Configuration for a single marker instance."""

    marker: Any = field(
        default=None,
        metadata={"help": "Marker object reference."}
    )
    scale: torch.Tensor = field(
        default=None,
        metadata={"help": "Marker scale tensor."}
    )


@dataclass
class IsaacLabPhysXParams(IsaacGymPhysXParams):
    """PhysX physics engine parameters with IsaacLab extensions."""

    gpu_found_lost_pairs_capacity: int = field(
        default=2**21,
        metadata={"help": "GPU capacity for found/lost collision pairs."}
    )
    gpu_max_rigid_contact_count: int = field(
        default=2**23,
        metadata={"help": "Maximum GPU rigid contact count."}
    )
    gpu_found_lost_aggregate_pairs_capacity: int = field(
        default=2**25,
        metadata={"help": "GPU capacity for aggregate found/lost pairs."}
    )
    gpu_max_rigid_patch_count: int = field(
        default=5 * 2**15,
        metadata={"help": "Maximum GPU rigid patch count."}
    )


@dataclass
class IsaacLabSimParams(SimParams):
    """PhysX-specific simulation parameters for IsaacLab."""

    physx: IsaacLabPhysXParams = field(
        default_factory=IsaacLabPhysXParams,
        metadata={"help": "PhysX engine parameters."}
    )


@dataclass
class IsaacLabSimulatorConfig(SimulatorConfig):
    """Configuration specific to IsaacLab simulator."""

    _target_: str = "protomotions.simulator.isaaclab.simulator.IsaacLabSimulator"
    w_last: bool = field(
        default=False,
        metadata={"help": "Quaternion format: False for wxyz (IsaacLab convention)."}
    )
    sim: IsaacLabSimParams = field(
        default_factory=IsaacLabSimParams,
        metadata={"help": "IsaacLab-specific simulation parameters."}
    )
