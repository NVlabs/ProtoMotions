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
from typing import Optional, List
from protomotions.envs.base_env.config import EnvConfig
from protomotions.envs.obs.config import MimicObsConfig, MaskedMimicObsConfig
from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
from protomotions.utils.config_builder import ConfigBuilder


@dataclass
class MimicEarlyTerminationEntry(ConfigBuilder):
    """Configuration for a single early termination criterion."""

    mimic_early_termination_key: str
    mimic_early_termination_thresh: float
    less_than: bool = True


@dataclass
class MimicEnvConfig(EnvConfig):
    """Configuration for mimic environment.

    Inherits reward_config from EnvConfig - all reward components are defined there.
    """

    _target_: str = "protomotions.envs.mimic.env.Mimic"

    mimic_obs: MimicObsConfig = field(default_factory=MimicObsConfig)

    # Mimic-specific termination and respawn params
    mimic_early_termination: Optional[List[MimicEarlyTerminationEntry]] = None
    mimic_bootstrap_on_episode_end: bool = True
    reset_grace_period: int = 5  # Grace period prevents early termination and zeroes out certain rewards during the first few steps after reset

    # Masked mimic configuration
    masked_mimic_obs: MaskedMimicObsConfig = field(default_factory=MaskedMimicObsConfig)

    # Mimic-specific motion manager configuration
    motion_manager: MimicMotionManagerConfig = field(
        default_factory=MimicMotionManagerConfig
    )

    # To allow kinematic playback
    sync_motion: bool = False
