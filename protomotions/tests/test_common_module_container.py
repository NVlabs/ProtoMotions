# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for generic TensorDict module containers."""

from types import SimpleNamespace

import torch
import pytest
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from protomotions.agents.base_agent.model import (
    BaseModel,
    ProtoMotionsTensorDictModule,
    RolloutStateSpec,
)
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


class _EpisodeNoiseModule(ProtoMotionsTensorDictModule):
    def __init__(self, config):
        super().__init__()
        self.in_keys = ["obs"]
        self.out_keys = ["latent"]

    def rollout_state_specs(self):
        return {"episode_noise": RolloutStateSpec(shape=(2,), init="normal")}

    def forward(self, tensordict: TensorDict):
        self.read_rollout_state(tensordict)
        tensordict["latent"] = tensordict["obs"] + tensordict["episode_noise"]
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


def test_saved_rollout_context_restores_nested_module_state_after_reseeding():
    container = ModuleContainer(
        ModuleContainerConfig(
            in_keys=["obs"],
            out_keys=["plain_out"],
            models=[
                ModuleContainerConfig(
                    _target_="protomotions.tests.test_common_module_container._EpisodeNoiseModule",
                ),
                ModuleContainerConfig(
                    _target_="protomotions.tests.test_common_module_container._PlainTensorDictModule",
                ),
            ],
        )
    )
    torch.manual_seed(0)
    container.reset_rollout_context(num_envs=3, device="cpu")
    noise = container.models[0].episode_noise
    before = noise.clone()

    saved = container.save_rollout_context()
    container.reset_rollout_context(env_ids=torch.tensor([0, 2]))
    assert not torch.equal(noise, before)
    container.load_rollout_context(saved)

    assert list(saved) == ["models.0.episode_noise"]
    assert torch.equal(noise, before)


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


# ---------------------------------------------------------------------------
# Module ordering: `models` is execution order, not a dependency graph
# ---------------------------------------------------------------------------


def _stub(in_keys, out_keys):
    """A config-shaped stand-in; find_key_flow_errors reads only in/out keys."""
    return SimpleNamespace(in_keys=list(in_keys), out_keys=list(out_keys))


def test_key_flow_accepts_a_correctly_ordered_list():
    config = ModuleContainerConfig(
        in_keys=["obs"],
        out_keys=["action"],
        models=[_stub(["obs"], ["token"]), _stub(["token"], ["action"])],
    )
    assert config.find_key_flow_errors() == []


def test_key_flow_reports_a_producer_that_runs_after_its_consumer():
    """The real defect: the producer was appended to the end of the list.

    A scene-conditioning channel was added by appending its normalizer and token
    MLP after the transformer that consumes the token. The names lined up, so it
    read as correct; the container validates in list order, so it was not.
    """
    config = ModuleContainerConfig(
        in_keys=["obs"],
        out_keys=["action"],
        models=[
            _stub(["obs", "scene_token"], ["action"]),  # consumer, first
            _stub(["scene_obs"], ["scene_norm"]),  # producers, appended after
            _stub(["scene_norm"], ["scene_token"]),
        ],
    )

    errors = config.find_key_flow_errors()

    assert len(errors) == 1
    assert "'scene_token'" in errors[0]
    assert "model 2" in errors[0]
    assert "execution order" in errors[0]


def test_key_flow_stays_quiet_about_keys_nothing_declares():
    """Some modules derive their real keys at construction, so silence is correct.

    DiscreteLatent takes its in_keys from its encoder rather than from config, so
    "no module declares this output" does not imply the key is missing. Only a
    known later producer is unambiguous, and only that is reported.
    """
    config = ModuleContainerConfig(
        in_keys=[],
        out_keys=["action"],
        models=[_stub(["produced_at_runtime"], ["action"])],
    )
    assert config.find_key_flow_errors() == []


def test_container_rejects_a_misordered_list_before_building_any_module():
    class _Exploding:
        def __init__(self, config):
            raise AssertionError("no module should be constructed")

    config = ModuleContainerConfig(
        in_keys=["obs"],
        out_keys=["action"],
        models=[
            SimpleNamespace(
                in_keys=["late_token"],
                out_keys=["action"],
                _target_=f"{__name__}._Exploding",
            ),
            SimpleNamespace(
                in_keys=["obs"],
                out_keys=["late_token"],
                _target_=f"{__name__}._Exploding",
            ),
        ],
    )

    with pytest.raises(AssertionError, match="out of order"):
        ModuleContainer(config)
