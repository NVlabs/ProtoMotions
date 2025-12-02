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
"""Motion imitation environment.

This package implements the Mimic environment for training full-body motion
tracking agents. It provides detailed rewards for matching reference motions
from a motion library.

Key Components:
    - Mimic: Motion imitation environment
    - MimicEnvConfig: Environment configuration
    - MimicObs: Motion tracking observations
    - MimicMotionManager: Reference motion management

Example:
    >>> from protomotions.envs.mimic.env import Mimic
    >>> env = Mimic(config, robot_config, simulator_config, device)
    >>> obs, info = env.reset()
    >>> next_obs, rewards, dones, info = env.step(actions)
"""
