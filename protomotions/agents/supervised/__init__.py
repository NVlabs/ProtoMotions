# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supervised imitation and distillation agents.

The package owns the generic supervised rollout loop and named student presets
such as MaskedMimic. Keep student architectures self-contained; share the loop
and key-based supervision losses, not architecture-specific inheritance.
"""
