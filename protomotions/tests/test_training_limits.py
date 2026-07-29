# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for whole-iteration training limits."""

import pytest

from protomotions.agents.base_agent.config import BaseAgentConfig


def test_training_max_iterations_takes_precedence():
    config = BaseAgentConfig(
        batch_size=15,
        training_max_steps=1,
        training_max_iterations=7,
        num_steps=3,
    )

    assert config.resolve_max_epochs(total_envs=5) == 7


def test_training_max_steps_keeps_existing_whole_iteration_floor():
    config = BaseAgentConfig(
        batch_size=15,
        training_max_steps=301,
        training_max_iterations=None,
        num_steps=3,
    )

    assert config.resolve_max_epochs(total_envs=5) == 20


def test_training_max_iterations_must_be_positive():
    config = BaseAgentConfig(
        batch_size=15,
        training_max_steps=300,
        training_max_iterations=0,
        num_steps=3,
    )

    with pytest.raises(ValueError, match="must be greater than zero"):
        config.resolve_max_epochs(total_envs=5)
