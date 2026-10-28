# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configured construction does not impose a new interface on child modules."""

from types import SimpleNamespace

import torch

from protomotions.agents.base_agent.model import ProtoMotionsTensorDictModule
from protomotions.agents.common.mlp import MLPWithConcat
from protomotions.agents.common.config import MLPWithConcatConfig


class PlainModule(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


def test_build_plain_module():
    config = SimpleNamespace(
        _target_="protomotions.tests.test_model_construction.PlainModule"
    )
    module = ProtoMotionsTensorDictModule._build_submodule(config)
    assert isinstance(module, PlainModule)
    assert module.config is config


def test_build_tensordict_module():
    config = MLPWithConcatConfig(in_keys=["obs"], out_keys=["action"], num_out=3)
    module = ProtoMotionsTensorDictModule._build_submodule(config)
    assert isinstance(module, MLPWithConcat)
    assert module.config is config
