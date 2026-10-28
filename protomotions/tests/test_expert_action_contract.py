# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from protomotions.agents.supervised.expert_utils import validate_expert_action_contract
from protomotions.envs.action.config import ActionConfig


def _scaled_action(action, scale, transform):
    return {
        "processed_action": action * scale,
        "stiffness_targets": action,
        "damping_targets": action,
    }


def _contract():
    robot = SimpleNamespace(
        kinematic_info=SimpleNamespace(dof_names=["left", "right"]),
        control={"stiffness": 10, "effort": 20},
    )
    env = SimpleNamespace(
        action_config=ActionConfig(
            fn=_scaled_action,
            params={"scale": torch.tensor([0.1, 0.2]), "transform": "clamp"},
        )
    )
    return robot, env


def test_identical_action_contract():
    robot, env = _contract()
    validate_expert_action_contract(robot, env, deepcopy(robot), deepcopy(env))


@pytest.mark.parametrize(
    "change", ["order", "gain", "effort", "scale", "transform", "missing"]
)
def test_rejects_incompatible_action_contract(change):
    robot, env = _contract()
    target, target_env = deepcopy(robot), deepcopy(env)
    if change == "order":
        target.kinematic_info.dof_names.reverse()
    elif change == "gain":
        target.control["stiffness"] = 11
    elif change == "effort":
        target.control["effort"] = 21
    elif change == "scale":
        target_env.action_config.params["scale"] *= 2
    elif change == "transform":
        target_env.action_config.params["transform"] = "tanh"
    else:
        target_env.action_config = None
    with pytest.raises(ValueError, match="Expert"):
        validate_expert_action_contract(robot, env, target, target_env)
