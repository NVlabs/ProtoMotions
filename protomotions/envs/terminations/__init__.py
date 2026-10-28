# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Termination compute kernels for environments.

Pure tensor functions (kernels) whose state inputs default to the context names
they read. A recipe imports a kernel from its defining module, binds it with
``MdpComponent.from_kernel`` and wraps it in a ``TerminationRule`` (bare for a
boolean kernel, with a threshold for a scalar one); this package re-exports
nothing.

Organized into:
- base: Boolean checks (contact fall, joint-limit blow-up) and the episode
  length cut the managers apply
- tracking: Scalar tracking errors (mean/max body position, rotation, anchor)
- task: Scalar task errors (path distance and height, steering speed and
  direction)
- low_pose, foot_tracking: reference-height-gated and foot-aligned limits
- rule: ``TerminationRule``, the one termination entry type
"""
