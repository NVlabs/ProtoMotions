# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for resolving trainer iteration limits."""


def resolve_training_max_epochs(config, total_envs: int, num_steps: int) -> int:
    """Resolve the whole-iteration training limit from an agent config."""
    training_max_iterations = getattr(config, "training_max_iterations", None)
    if training_max_iterations is not None:
        if training_max_iterations <= 0:
            raise ValueError("training_max_iterations must be greater than zero")
        return training_max_iterations

    return config.training_max_steps // total_envs // num_steps
