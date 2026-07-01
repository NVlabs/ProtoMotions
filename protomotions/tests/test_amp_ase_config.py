# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for AMP and ASE config defaults and validation."""

import pytest

from protomotions.agents.amp.config import (
    AMPAgentConfig,
    AMPModelConfig,
    AMPParametersConfig,
    DiscriminatorConfig,
)
from protomotions.agents.ase.config import (
    ASEAgentConfig,
    ASEDiscriminatorEncoderConfig,
    ASEModelConfig,
    ASEParametersConfig,
)
from protomotions.agents.common.config import ModuleContainerConfig
from protomotions.agents.ppo.config import PPOActorConfig


def _actor_config():
    return PPOActorConfig(mu_key="mu")


def test_amp_parameters_and_model_defaults_are_independent():
    params = AMPParametersConfig()
    first_model = AMPModelConfig(actor=_actor_config())
    second_model = AMPModelConfig(actor=_actor_config())

    first_model.discriminator.out_keys.append("extra")

    assert params.discriminator_reward_w == 1.0
    assert params.discriminator_grad_penalty == 5.0
    assert params.use_disc_critic is True
    assert first_model._target_ == "protomotions.agents.amp.model.AMPModel"
    assert first_model.discriminator._target_ == "protomotions.agents.amp.model.Discriminator"
    assert first_model.discriminator_optimizer.lr == 1e-4
    assert first_model.disc_critic_optimizer.lr == 1e-4
    assert second_model.discriminator.out_keys == ["disc_logits"]


def test_amp_agent_config_wires_amp_model_and_reference_components():
    model = AMPModelConfig(actor=_actor_config())
    config = AMPAgentConfig(
        batch_size=32,
        training_max_steps=1000,
        model=model,
    )

    assert config._target_ == "protomotions.agents.amp.agent.AMP"
    assert isinstance(config.model, AMPModelConfig)
    assert config.reference_obs_components == {}
    assert isinstance(config.model.disc_critic, ModuleContainerConfig)


def test_ase_discriminator_encoder_requires_encoder_output_size():
    with pytest.raises(AssertionError, match="encoder_out_size"):
        ASEDiscriminatorEncoderConfig()

    config = ASEDiscriminatorEncoderConfig(
        encoder_out_size=8,
        in_keys=["disc_obs"],
    )

    assert isinstance(config, DiscriminatorConfig)
    assert config._target_ == "protomotions.agents.ase.model.ASEDiscriminatorEncoder"
    assert config.in_keys == ["disc_obs"]
    assert config.out_keys == ["disc_logits", "mi_enc_output"]


def test_ase_parameters_model_and_agent_defaults():
    params = ASEParametersConfig()
    model = ASEModelConfig(actor=_actor_config())
    agent = ASEAgentConfig(
        batch_size=16,
        training_max_steps=200,
        model=ASEModelConfig(actor=_actor_config()),
    )

    assert params.latent_dim == 64
    assert params.latent_steps_min == 1
    assert params.latent_steps_max == 150
    assert params.mi_reward_w == 0.5
    assert model._target_ == "protomotions.agents.ase.model.ASEModel"
    assert isinstance(model.mi_critic, ModuleContainerConfig)
    assert model.mi_critic_optimizer.lr == 1e-4
    assert agent._target_ == "protomotions.agents.ase.agent.ASE"
    assert isinstance(agent.model, ASEModelConfig)
    assert isinstance(agent.ase_parameters, ASEParametersConfig)


def test_prior_peft_amp_config_explicitly_combines_peft_and_amp_defaults():
    from protomotions.agents.peft.prior_amp_config import (
        DiscretePriorPEFTRLFTAMPAgentConfig,
        DiscretePriorPEFTRLFTAMPModelConfig,
    )
    from protomotions.agents.peft.prior_config import DiscretePriorPEFTActorConfig

    model = DiscretePriorPEFTRLFTAMPModelConfig(actor=DiscretePriorPEFTActorConfig())
    agent = DiscretePriorPEFTRLFTAMPAgentConfig(
        batch_size=8,
        training_max_steps=32,
        model=model,
    )

    assert agent._target_ == (
        "protomotions.agents.peft.prior_amp_agent."
        "DiscretePriorPEFTRLFTAMPAgent"
    )
    assert model._target_ == (
        "protomotions.agents.peft.prior_amp_model."
        "DiscretePriorPEFTRLFTAMPModel"
    )
    assert isinstance(agent.amp_parameters, AMPParametersConfig)
    assert isinstance(model.discriminator, DiscriminatorConfig)
    assert isinstance(model.disc_critic, ModuleContainerConfig)
    assert model.discriminator_optimizer.lr == 1e-4
    assert model.disc_critic_optimizer.lr == 1e-4
