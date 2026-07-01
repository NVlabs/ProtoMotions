# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for frozen pretrained module loading helpers."""

from types import SimpleNamespace

import torch
from tensordict import TensorDict

from protomotions.agents.base_agent.config import BaseModelConfig
from protomotions.agents.base_agent.model import BaseModel
from protomotions.agents.common.config import PretrainedModelConfig
from protomotions.agents.common import pretrained as pretrained_module
from protomotions.agents.common.pretrained import load_pretrained_model_module


class _LazyCheckpointModel(BaseModel):
    def __init__(self, config):
        super().__init__(BaseModelConfig())
        self.linear = torch.nn.LazyLinear(1, bias=False)
        self.materialized_from_state = False

    def materialize_from_state_dict(self, state_dict):
        in_features = state_dict["linear.weight"].shape[1]
        self.linear(torch.zeros(1, in_features))
        self.materialized_from_state = True

    def forward(self, tensordict: TensorDict, log_internals: bool = False):
        tensordict["action"] = self.linear(tensordict["obs"])
        return tensordict


def test_load_pretrained_model_module_materializes_lazy_modules_from_state_dict(
    tmp_path,
    monkeypatch,
):
    checkpoint_path = tmp_path / "lazy.pt"
    torch.save(
        {"model": {"linear.weight": torch.tensor([[2.0, 3.0]])}},
        checkpoint_path,
    )
    monkeypatch.setattr(
        pretrained_module,
        "load_resolved_configs_from_checkpoint",
        lambda *args, **kwargs: {
            "agent": SimpleNamespace(
                model=SimpleNamespace(
                    _target_="protomotions.tests.test_common_pretrained._LazyCheckpointModel"
                )
            )
        },
    )

    model = load_pretrained_model_module(
        PretrainedModelConfig(
            checkpoint_path=str(checkpoint_path),
            module_path="",
            freeze=False,
        ),
        device=torch.device("cpu"),
    )

    assert model.materialized_from_state is True
    assert torch.equal(model.linear.weight, torch.tensor([[2.0, 3.0]]))
