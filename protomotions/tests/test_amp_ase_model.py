# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for AMP and ASE TensorDict model components."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from protomotions.agents.amp.config import AMPModelConfig, DiscriminatorConfig
from protomotions.agents.amp.model import AMPModel, Discriminator
from protomotions.agents.ase.config import (
    ASEDiscriminatorEncoderConfig,
    ASEModelConfig,
)
from protomotions.agents.ase.model import ASEDiscriminatorEncoder, ASEModel
from protomotions.agents.common.config import (
    MLPWithConcatConfig,
    ModuleContainerConfig,
    ObsProcessorConfig,
)
from protomotions.agents.ppo.config import PPOActorConfig


class _LinearHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_keys = config.in_keys
        self.out_keys = config.out_keys
        self.weight = nn.Parameter(
            torch.full(
                (config.num_out, config.in_dim),
                getattr(config, "weight_value", 1.0),
            )
        )
        self.bias = nn.Parameter(
            torch.full((config.num_out,), getattr(config, "bias_value", 0.0))
        )
        self.mlp = [self]

    def forward(self, tensordict):
        tensordict[self.out_keys[0]] = (
            tensordict[self.in_keys[0]] @ self.weight.t() + self.bias
        )
        return tensordict


class _NoConfigModule(nn.Module):
    in_keys = ["raw_obs"]
    out_keys = ["disc_logits"]

    def __init__(self, config):
        super().__init__()

    def forward(self, tensordict):
        tensordict["disc_logits"] = tensordict["raw_obs"].sum(dim=-1, keepdim=True)
        return tensordict


def _head_config(in_key, out_key, in_dim, num_out, weight_value=1.0):
    return SimpleNamespace(
        _target_="protomotions.tests.test_amp_ase_model._LinearHead",
        in_keys=[in_key],
        out_keys=[out_key],
        in_dim=in_dim,
        num_out=num_out,
        weight_value=weight_value,
    )


def _mlp(in_keys, out_key, num_out, normalize_obs=False):
    return MLPWithConcatConfig(
        in_keys=in_keys,
        out_keys=[out_key],
        num_out=num_out,
        layers=[],
        normalize_obs=normalize_obs,
        norm_clamp_value=100.0,
    )


def _actor_config():
    return PPOActorConfig(
        mu_key="mu",
        in_keys=["obs"],
        out_keys=["action", "mean_action", "neglogp"],
        mu_model=_mlp(["obs"], "mu", 1),
        num_out=1,
        actor_logstd=-2.0,
    )


def _amp_config(out_keys=None, in_keys=None):
    return AMPModelConfig(
        in_keys=in_keys or ["obs", "disc_obs"],
        out_keys=out_keys
        or [
            "action",
            "mean_action",
            "neglogp",
            "value",
            "disc_logits",
            "disc_value",
        ],
        actor=_actor_config(),
        critic=ModuleContainerConfig(
            in_keys=["obs"],
            out_keys=["value"],
            models=[_mlp(["obs"], "value", 1)],
        ),
        discriminator=DiscriminatorConfig(
            in_keys=["disc_obs"],
            out_keys=["disc_logits"],
            models=[_mlp(["disc_obs"], "disc_logits", 1)],
        ),
        disc_critic=ModuleContainerConfig(
            in_keys=["disc_obs"],
            out_keys=["disc_value"],
            models=[_mlp(["disc_obs"], "disc_value", 1)],
        ),
    )


def _ase_discriminator_config():
    return ASEDiscriminatorEncoderConfig(
        encoder_out_size=2,
        in_keys=["disc_obs"],
        out_keys=["disc_logits", "mi_enc_output"],
        models=[
            _head_config("disc_obs", "trunk", in_dim=2, num_out=2, weight_value=0.5),
            ModuleContainerConfig(
                in_keys=["trunk"],
                out_keys=["disc_logits", "mi_enc_output"],
                models=[
                    _mlp(["trunk"], "disc_logits", 1),
                    _head_config(
                        "trunk",
                        "mi_enc_output",
                        in_dim=2,
                        num_out=2,
                        weight_value=2.0,
                    ),
                ],
            ),
        ],
    )


def _ase_config(out_keys=None, in_keys=None):
    return ASEModelConfig(
        in_keys=in_keys or ["obs", "disc_obs", "latents"],
        out_keys=out_keys
        or [
            "action",
            "mean_action",
            "neglogp",
            "value",
            "disc_logits",
            "mi_enc_output",
            "disc_value",
            "mi_value",
        ],
        actor=_actor_config(),
        critic=ModuleContainerConfig(
            in_keys=["obs"],
            out_keys=["value"],
            models=[_mlp(["obs"], "value", 1)],
        ),
        discriminator=_ase_discriminator_config(),
        disc_critic=ModuleContainerConfig(
            in_keys=["disc_obs"],
            out_keys=["disc_value"],
            models=[_mlp(["disc_obs"], "disc_value", 1)],
        ),
        mi_critic=ModuleContainerConfig(
            in_keys=["latents"],
            out_keys=["mi_value"],
            models=[_mlp(["latents"], "mi_value", 1)],
        ),
    )


def _batch():
    return TensorDict(
        {
            "obs": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "disc_obs": torch.tensor([[0.5, 1.0], [1.5, 2.0]]),
            "latents": torch.tensor([[1.0, 0.0], [-1.0, 0.5]]),
        },
        batch_size=2,
    )


def test_discriminator_discovers_effective_gradient_penalty_keys_and_rewards():
    config = DiscriminatorConfig(
        in_keys=["raw_obs", "mask"],
        out_keys=["disc_logits"],
        models=[
            ObsProcessorConfig(
                in_keys=["raw_obs"],
                out_keys=["processed_obs"],
            ),
            _mlp(["processed_obs"], "disc_logits", 1, normalize_obs=True),
        ],
    )
    discriminator = Discriminator(config)
    discriminator.eval()

    out = discriminator(
        TensorDict(
            {
                "raw_obs": torch.ones(2, 2),
                "mask": torch.tensor([[True], [False]]),
            },
            batch_size=2,
        )
    )
    logits = torch.tensor([[-1.0], [0.0], [2.0]])

    assert discriminator._grad_penalty_keys == [
        "processed_obs",
        "norm_processed_obs",
        "mask",
    ]
    assert out["disc_logits"].shape == (2, 1)
    assert torch.allclose(
        discriminator.compute_disc_reward(logits),
        -torch.log(torch.clamp(1 - torch.sigmoid(logits), min=1e-4)),
    )
    assert discriminator.all_discriminator_weights() == discriminator.logit_weights()


def test_discriminator_falls_back_to_config_inputs_when_no_targets_discovered():
    discriminator = Discriminator(
        DiscriminatorConfig(
            in_keys=["raw_obs"],
            out_keys=["disc_logits"],
            models=[
                SimpleNamespace(
                    _target_="protomotions.tests.test_amp_ase_model._NoConfigModule"
                )
            ],
        )
    )
    td = TensorDict({"raw_obs": torch.ones(2, 3)}, batch_size=2)

    out = discriminator(td)

    assert discriminator._grad_penalty_keys == ["raw_obs"]
    assert torch.equal(out["disc_logits"], torch.ones(2, 1) * 3)


def test_amp_model_forwards_actor_critic_discriminator_and_disc_critic():
    torch.manual_seed(0)
    model = AMPModel(_amp_config())
    td = _batch().select("obs", "disc_obs")

    out = model(td)

    assert out["action"].shape == (2, 1)
    assert out["mean_action"].shape == (2, 1)
    assert out["neglogp"].shape == (2,)
    assert out["value"].shape == (2, 1)
    assert out["disc_logits"].shape == (2, 1)
    assert out["disc_value"].shape == (2, 1)


def test_amp_model_validates_discriminator_keys_against_model_contract():
    with pytest.raises(AssertionError, match="Discriminator output key"):
        AMPModel(
            _amp_config(
                out_keys=["action", "mean_action", "neglogp", "value", "disc_value"]
            )
        )

    with pytest.raises(AssertionError, match="Discriminator input key"):
        AMPModel(_amp_config(in_keys=["obs"]))


def test_ase_discriminator_encoder_initializes_encoder_and_partitions_weights():
    model = ASEDiscriminatorEncoder(_ase_discriminator_config())
    td = _batch().select("disc_obs")
    out = model(td)
    trunk = model.models[0]
    disc_head = model.models[1].models[0]
    disc_logit_weight = disc_head.mlp[-1].weight
    encoder_head = model.models[1].models[1]

    assert out["disc_logits"].shape == (2, 1)
    assert out["mi_enc_output"].shape == (2, 2)
    assert model._encoder_initialized is True
    assert torch.all(encoder_head.weight <= 0.1)
    assert torch.all(encoder_head.weight >= -0.1)
    assert torch.equal(encoder_head.bias, torch.zeros_like(encoder_head.bias))
    assert [id(weight) for weight in model.all_weights()] == [
        id(trunk.weight),
        id(disc_logit_weight),
        id(encoder_head.weight),
    ]
    assert [id(weight) for weight in model.all_discriminator_weights()] == [
        id(trunk.weight),
        id(disc_logit_weight),
    ]
    assert [id(weight) for weight in model.logit_weights()] == [id(disc_logit_weight)]
    assert [id(weight) for weight in model.all_enc_weights()] == [
        id(trunk.weight),
        id(encoder_head.weight),
    ]
    assert [id(weight) for weight in model.enc_weights()] == [id(encoder_head.weight)]
    assert [id(weight) for weight in model._get_weights_from_module(model.models[1])] == [
        id(disc_logit_weight),
        id(encoder_head.weight),
    ]


def test_ase_discriminator_encoder_computes_mutual_information_rewards():
    model = ASEDiscriminatorEncoder(_ase_discriminator_config())
    td = TensorDict(
        {
            "mi_enc_output": torch.tensor([[0.5, 0.5], [-1.0, 0.5]]),
            "latents": torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
        },
        batch_size=2,
    )

    shifted = model.compute_mi_reward(td, mi_hypersphere_reward_shift=True)
    unshifted = model.compute_mi_reward(td, mi_hypersphere_reward_shift=False)

    assert torch.allclose(
        model.calc_von_mises_fisher_enc_error(td["mi_enc_output"], td["latents"]),
        torch.tensor([[-0.5], [0.5]]),
    )
    assert torch.allclose(shifted, torch.tensor([[0.75], [0.25]]))
    assert torch.allclose(unshifted, torch.tensor([[0.5], [0.0]]))


def test_ase_model_forwards_amp_outputs_and_mi_critic():
    torch.manual_seed(0)
    model = ASEModel(_ase_config())

    out = model(_batch())

    assert out["action"].shape == (2, 1)
    assert out["value"].shape == (2, 1)
    assert out["disc_logits"].shape == (2, 1)
    assert out["mi_enc_output"].shape == (2, 2)
    assert out["disc_value"].shape == (2, 1)
    assert out["mi_value"].shape == (2, 1)


def test_ase_model_forwards_with_internal_logging_enabled():
    torch.manual_seed(0)
    model = ASEModel(_ase_config())

    out = model(_batch(), log_internals=True)

    assert out["disc_logits"].shape == (2, 1)
    assert out["mi_enc_output"].shape == (2, 2)
    assert out["mi_value"].shape == (2, 1)


def test_ase_model_validates_mi_critic_keys_against_model_contract():
    with pytest.raises(AssertionError, match="MI critic output key"):
        ASEModel(
            _ase_config(
                out_keys=[
                    "action",
                    "mean_action",
                    "neglogp",
                    "value",
                    "disc_logits",
                    "mi_enc_output",
                    "disc_value",
                ]
            )
        )

    with pytest.raises(AssertionError, match="MI critic input key"):
        ASEModel(_ase_config(in_keys=["obs", "disc_obs"]))
