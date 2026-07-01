# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for generic TensorDict module containers."""

import torch
import pytest
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from protomotions.agents.base_agent.model import BaseModel, ProtoMotionsTensorDictModule
from protomotions.agents.common.common import ModuleContainer
from protomotions.agents.common.config import ModuleContainerConfig


class _StatefulModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.in_keys = ["obs"]
        self.out_keys = ["latent"]

    def rollout_context_keys(self) -> list:
        return ["latent_noise"]

    def forward(self, tensordict: TensorDict, log_internals: bool = False):
        tensordict["latent"] = tensordict["obs"] + 1.0
        tensordict["latent_noise"] = tensordict["obs"] * 0.0 + 2.0
        return tensordict


class _ResettableModel(_StatefulModel):
    def __init__(self, config):
        super().__init__(config)
        self.reset_calls = []

    def reset_rollout_context(
        self, env_ids=None, num_envs: int = None, device=None
    ) -> None:
        self.reset_calls.append((env_ids, num_envs, device))


class _CommonStateModule(ProtoMotionsTensorDictModule):
    def __init__(self, config):
        super().__init__()
        self.in_keys = ["latent"]
        self.out_keys = ["common_out"]

    def rollout_context_keys(self) -> list:
        return ["common_state"]

    def forward(self, tensordict: TensorDict):
        tensordict["common_out"] = tensordict["latent"] * 4.0
        tensordict["common_state"] = tensordict["latent"] * 0.0 + 5.0
        return tensordict


class _PlainTensorDictModule(TensorDictModuleBase):
    def __init__(self, config):
        super().__init__()
        self.in_keys = ["latent"]
        self.out_keys = ["plain_out"]

    def forward(self, tensordict: TensorDict):
        tensordict["plain_out"] = tensordict["latent"] * 3.0
        return tensordict


class _BaseContractModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.in_keys = ["obs"]
        self.out_keys = ["action"]

    def forward(self, tensordict: TensorDict, log_internals: bool = False):
        tensordict["action"] = tensordict["obs"] + 1.0
        return tensordict


def _container():
    return ModuleContainer(
        ModuleContainerConfig(
            in_keys=["obs"],
            out_keys=["plain_out"],
            models=[
                ModuleContainerConfig(
                    _target_="protomotions.tests.test_common_module_container._StatefulModel",
                ),
                ModuleContainerConfig(
                    _target_="protomotions.tests.test_common_module_container._PlainTensorDictModule",
                ),
            ],
        )
    )


def test_module_container_forwards_rollout_context_keys_from_base_models_only():
    container = _container()

    assert container.rollout_context_keys() == ["latent_noise"]


def test_module_container_forwards_common_rollout_context_keys():
    container = ModuleContainer(
        ModuleContainerConfig(
            in_keys=["obs"],
            out_keys=["common_out"],
            models=[
                ModuleContainerConfig(
                    _target_="protomotions.tests.test_common_module_container._StatefulModel",
                ),
                ModuleContainerConfig(
                    _target_="protomotions.tests.test_common_module_container._CommonStateModule",
                ),
            ],
        )
    )
    td = TensorDict(
        {
            "obs": torch.tensor([[1.0], [2.0]]),
        },
        batch_size=2,
    )

    out = container(td)

    assert container.rollout_context_keys() == ["latent_noise", "common_state"]
    assert torch.equal(out["common_out"], torch.tensor([[8.0], [12.0]]))
    assert torch.equal(out["common_state"], torch.tensor([[5.0], [5.0]]))


def test_module_container_propagates_reset_to_internal_common_modules():
    container = ModuleContainer(
        ModuleContainerConfig(
            in_keys=["obs"],
            out_keys=["plain_out"],
            models=[
                ModuleContainerConfig(
                    _target_="protomotions.tests.test_common_module_container._ResettableModel",
                ),
                ModuleContainerConfig(
                    _target_="protomotions.tests.test_common_module_container._PlainTensorDictModule",
                ),
            ],
        )
    )
    env_ids = torch.tensor([0, 2])

    container.reset_rollout_context(env_ids=env_ids, num_envs=3, device="cpu")

    assert container.models[0].reset_calls == [(env_ids, 3, "cpu")]


def test_base_model_experience_buffer_keys_include_rollout_context_keys_once():
    model = _StatefulModel(config=None)
    model.out_keys = ["action", "latent_noise"]

    assert model.experience_buffer_keys() == ["action", "latent_noise"]


def test_base_model_default_hooks_are_generic_only():
    model = _BaseContractModel(config=None)
    td = TensorDict({"obs": torch.tensor([[2.0]])}, batch_size=1)

    materialized = model.materialize(td.clone())

    assert torch.equal(materialized["action"], torch.tensor([[3.0]]))
    assert model.optimization_module() is model
    assert model.materialize_from_state_dict({"unused": torch.ones(())}) is None
    assert not hasattr(model, "collect_rollout")
    assert not hasattr(model, "collect_expert_rollout")


def test_module_container_rejects_unavailable_model_inputs():
    with pytest.raises(AssertionError, match="requires .*latent"):
        ModuleContainer(
            ModuleContainerConfig(
                in_keys=["obs"],
                out_keys=["plain_out"],
                models=[
                    ModuleContainerConfig(
                        _target_="protomotions.tests.test_common_module_container._PlainTensorDictModule",
                    ),
                ],
            )
        )


def test_module_container_rejects_promised_outputs_not_produced():
    with pytest.raises(AssertionError, match="promises out_key"):
        ModuleContainer(
            ModuleContainerConfig(
                in_keys=["obs"],
                out_keys=["missing_out"],
                models=[
                    ModuleContainerConfig(
                        _target_="protomotions.tests.test_common_module_container._BaseContractModel",
                    ),
                ],
            )
        )
