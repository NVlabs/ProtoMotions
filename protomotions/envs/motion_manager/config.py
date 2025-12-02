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

from dataclasses import dataclass
from typing import Optional, List, Union
from protomotions.utils.config_builder import ConfigBuilder


@dataclass
class MotionManagerConfig(ConfigBuilder):
    """Configuration for motion management."""

    # Example: Use specific motion IDs during evaluation
    # subset_method: [0, 1, 2, 3, 4]  # Use specific motions (length must equal num_envs)
    # subset_method: "first"  # Use first N motions where N = num_envs

    # Example: Exclude specific motion IDs from probabilistic sampling
    # exclude_motion_ids: [2, 5, 8, 12]  # Exclude these motions from sampling

    _target_: str = "protomotions.envs.motion_manager.motion_manager.MotionManager"

    # By default, without dynamic sampling the motion manager picks a random motion
    # By default, this sets 20% chance to sample an initial pose.
    # Especially for AMP this helps prevent the agent from immediately getting stuck in a local-minima.
    init_start_prob: float = 0.2

    # Motion subset configuration for evaluation/testing
    # Can be:
    # - "first": use first N motions (based on num_envs)
    # - "last": use last N motions
    # - "random": randomly sample N motions
    # - List of motion indices: [0, 1, 5, 10] to use specific motions (length must equal num_envs)
    # - null: use all motions (default)
    subset_method: Optional[Union[str, List[int]]] = None

    # Motion exclusion configuration for training/evaluation
    # Excludes specific motion IDs from probabilistic sampling by setting their weights to 0 before each sampling
    # Persistent exclusion even if motion weights are updated by other classes during training
    # Can be:
    # - List of motion indices: [2, 5, 8] to exclude these specific motions from sampling
    # - null: don't exclude any motions (default)
    exclude_motion_ids: Optional[List[int]] = None

    # Realign the motion with the humanoid on each step
    # This is useful when the motion is not perfectly retargeted to the humanoid
    # preventing tracking errors from accumulating over time
    realign_motion_with_humanoid_on_each_step: bool = False


@dataclass
class MimicMotionManagerConfig(MotionManagerConfig):
    """Configuration for mimic motion management."""

    _target_: str = (
        "protomotions.envs.motion_manager.mimic_motion_manager.MimicMotionManager"
    )

    resample_on_reset: bool = True
