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
        "load_experiment_configs_from_checkpoint",
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


def test_load_pretrained_prefers_the_checkpoint_when_both_are_set(
    tmp_path, monkeypatch
):
    """A present checkpoint_path wins over an embedded module_config.

    The GPC bundle carries both at train time (the teacher checkpoint for its
    weights, the embedded config so inference is self-contained once the path is
    blanked). Training must load the teacher's trained weights, not build a fresh
    module from the embedded config, so the frozen decoder is not random.
    """
    checkpoint_path = tmp_path / "teacher.pt"
    torch.save(
        {"model": {"linear.weight": torch.tensor([[2.0, 3.0]])}}, checkpoint_path
    )
    monkeypatch.setattr(
        pretrained_module,
        "load_experiment_configs_from_checkpoint",
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
            module_config=SimpleNamespace(
                _target_="protomotions.tests.test_common_pretrained._LazyCheckpointModel"
            ),
            freeze=False,
        ),
        device=torch.device("cpu"),
    )

    # The checkpoint branch ran (weights loaded from state_dict), not the
    # module_config branch (which builds an uninitialized module).
    assert model.materialized_from_state is True
    assert torch.equal(model.linear.weight, torch.tensor([[2.0, 3.0]]))


def test_load_pretrained_raises_on_a_set_but_missing_checkpoint(tmp_path):
    """A set-but-missing checkpoint_path raises, even with an embedded config.

    Only a blank path (inference blanks it when a config is embedded) falls back
    to module_config. A mistyped or moved teacher checkpoint must fail loudly
    rather than silently build untrained weights from the embedded config.
    """
    import pytest

    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        load_pretrained_model_module(
            PretrainedModelConfig(
                checkpoint_path=str(tmp_path / "does_not_exist.ckpt"),
                module_path="",
                module_config=SimpleNamespace(
                    _target_="protomotions.tests.test_common_pretrained._LazyCheckpointModel"
                ),
                freeze=False,
            ),
            device=torch.device("cpu"),
        )
