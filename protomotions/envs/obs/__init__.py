# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observation utilities and compute kernels for environments.

Contains pure tensor compute kernels for:
- Humanoid observations (max-coords, reduced-coords, historical)
- Mimic target poses
- Masked mimic observations
- Steering observations
- Path following observations
- State history buffers
- Observation noise utilities

Use MdpComponent in experiment configs to bind kernels to context paths.
"""
