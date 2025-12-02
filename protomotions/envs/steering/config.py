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
from protomotions.envs.obs.config import SteeringObsConfig


@dataclass
class SteeringEnvConfig(EnvConfig):
    """Configuration for steering environment.

    The steering task trains agents to walk in a target direction at a target speed.
    Target direction and speed change periodically during training.

    All task parameters (speed range, heading change frequency, etc.) are configured
    via the steering_obs field.
    """

    _target_: str = "protomotions.envs.steering.env.Steering"

    # Override default height termination to True for steering
    enable_height_termination: bool = True

    # Steering observations and task parameters (enabled by default for steering env)
    steering_obs: SteeringObsConfig = field(
        default_factory=lambda: SteeringObsConfig(enabled=True)
    )
