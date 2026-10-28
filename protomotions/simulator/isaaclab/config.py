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
    gpu_collision_stack_size: int = field(
        default=2**26,
        metadata={
            "help": (
                "PhysX collision stack in bytes (IsaacLab's default, 2**26 = 64 MiB). "
                "When it is too small PhysX logs 'collisionStackSize buffer overflow "
                "... Contacts have been dropped' with the size it needs. Large "
                "triangle-mesh scenes need more (SAMP at 15,000 envs used 2**30). "
                "Keep it at or below 2**30: the value is written to USD as a 32-bit int. "
                "On the command line write the number out (1073741824, not 2**30)."
            )
        },
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
        default=True,
        metadata={"help": "Quaternion format: True for xyzw (IsaacLab 3 convention)."}
    )
    sim: IsaacLabSimParams = field(
        default_factory=IsaacLabSimParams,
        metadata={"help": "IsaacLab-specific simulation parameters."}
    )
