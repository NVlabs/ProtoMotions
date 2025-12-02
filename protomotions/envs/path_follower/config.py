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
from protomotions.envs.base_env.config import EnvConfig
from protomotions.envs.obs.config import PathObsConfig


@dataclass
class PathFollowerEnvConfig(EnvConfig):
    """Configuration for path following environment.

    The path following task trains agents to follow predefined trajectories.
    Target waypoints are sampled from the path generator and provided as observations.

    All task parameters (path generation, termination conditions, etc.) are configured
    via the path_obs field.
    """

    _target_: str = "protomotions.envs.path_follower.env.PathFollowing"

    # Path observations and task parameters (enabled by default for path follower env)
    path_obs: PathObsConfig = field(default_factory=lambda: PathObsConfig(enabled=True))
