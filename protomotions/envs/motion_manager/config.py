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
"""Configuration classes for motion manager components.

This module contains all configuration dataclasses for motion manager functionality,
co-located with the motion manager implementations in the same directory.
"""

from typing import Optional, List, Union
from dataclasses import dataclass, field


@dataclass
class MotionManagerConfig:
    """Configuration for motion management."""

    _target_: str = "protomotions.envs.motion_manager.motion_manager.MotionManager"

    init_start_prob: float = field(
        default=0.2,
        metadata={
            "help": "Probability to sample an initial pose instead of random time. Helps prevent local-minima in AMP.",
            "min": 0.0,
            "max": 1.0,
        }
    )

    subset_method: Optional[Union[str, List[int]]] = field(
        default=None,
        metadata={
            "help": "Motion subset for evaluation: 'first', 'last', 'random', or list of motion IDs. None uses all motions.",
            "options": ["first", "last", "random"],
        }
    )

    exclude_motion_ids: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "Motion IDs to exclude from sampling. Useful for removing problematic motions.",
        }
    )

    exclude_motions_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to file with motion IDs to exclude (one per line). Can also be an expert training directory.",
        }
    )

    realign_motion_with_humanoid_on_each_step: bool = field(
        default=False,
        metadata={
            "help": "Realign motion with humanoid each step. Prevents tracking error accumulation for imperfect retargeting.",
        }
    )


@dataclass
class MimicMotionManagerConfig(MotionManagerConfig):
    """Configuration for mimic motion management."""

    _target_: str = (
        "protomotions.envs.motion_manager.mimic_motion_manager.MimicMotionManager"
    )

    resample_on_reset: bool = field(
        default=True,
        metadata={"help": "Whether to resample motion on environment reset."}
    )
