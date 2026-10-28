# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reward compute kernels for environments.

Pure tensor functions (kernels) whose state inputs default to the context names
they read. A recipe imports a kernel from its defining module and binds it with
``MdpComponent.from_kernel(kernel, meta=RewardMeta(...), **overrides)``; this
package re-exports nothing.

Organized into:
- base: Primitive functions (MSE, norm, rotation error, power consumption)
- tracking: Motion imitation rewards (AMP/DeepMimic, BeyondMimic style)
- task: Task-specific rewards (steering, path following, target reaching)
- regularization: Regularization penalties (action smoothness, limits, contacts)
- meta: ``RewardMeta``, the per-reward weight and clamps
- combine: how the weighted rewards are summed
"""
