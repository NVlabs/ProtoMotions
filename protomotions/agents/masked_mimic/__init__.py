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
"""MaskedMimic implementation for versatile motion control.

This package implements the MaskedMimic algorithm which learns to reconstruct
expert tracker actions from partial observations. Trains on data from a full-body
motion tracker while randomly masking observations.

Key Components:
    - MaskedMimic: Main MaskedMimic agent
    - MaskedMimicModel: Model with optional VAE
    - MaskedMimicAgentConfig: Configuration

Training Process:
    1. Phase 1: Train expert full-body tracker (separate)
    2. Phase 2: Train MaskedMimic to imitate expert with masked observations

Example:
    >>> from protomotions.agents.masked_mimic.agent import MaskedMimic
    >>> config.expert_model_path = "results/expert_tracker/"
    >>> agent = MaskedMimic(fabric, env, config)
    >>> agent.train()

Reference:
    Tessler et al. "MaskedMimic: Unified Physics-Based Character Control Through Masked Motion Inpainting" (2024)
"""
