# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
