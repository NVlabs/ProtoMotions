# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Newton must not park evaluation envs.

Parking (teleport to z<<0) exists to relieve PhysX's broadphase pair budget.
Newton replicates environments as separate worlds, so there is no cross-env
broadphase to reduce. A parked robot still free-falls with PD control active:
joint velocities diverge and the resulting non-finite values persist in solver
warm-start memory, permanently poisoning the parked envs. NewtonSimulator
therefore overrides park_envs as a documented no-op; these tests pin that
contract.
"""

from unittest.mock import MagicMock

import pytest
import torch


def _newton_simulator_cls():
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    from protomotions.simulator.newton.simulator import NewtonSimulator

    return NewtonSimulator


def test_park_envs_is_overridden():
    from protomotions.simulator.base_simulator.simulator import Simulator

    NewtonSimulator = _newton_simulator_cls()
    assert NewtonSimulator.park_envs is not Simulator.park_envs


def test_park_envs_touches_no_simulator_state():
    NewtonSimulator = _newton_simulator_cls()
    mock_sim = MagicMock()
    env_ids = torch.arange(4)

    result = NewtonSimulator.park_envs(mock_sim, env_ids)

    assert result is None
    assert mock_sim.mock_calls == []
