# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for whole-iteration training limits."""

from types import SimpleNamespace

import pytest

from protomotions.agents.base_agent.training_limits import resolve_training_max_epochs


def test_training_max_iterations_takes_precedence():
    config = SimpleNamespace(
        training_max_steps=1,
        training_max_iterations=7,
    )

    assert resolve_training_max_epochs(config, total_envs=5, num_steps=3) == 7


def test_training_max_steps_keeps_existing_whole_iteration_floor():
    config = SimpleNamespace(
        training_max_steps=301,
        training_max_iterations=None,
    )

    assert resolve_training_max_epochs(config, total_envs=5, num_steps=3) == 20


def test_training_max_iterations_must_be_positive():
    config = SimpleNamespace(
        training_max_steps=300,
        training_max_iterations=0,
    )

    with pytest.raises(ValueError, match="must be greater than zero"):
        resolve_training_max_epochs(config, total_envs=5, num_steps=3)
