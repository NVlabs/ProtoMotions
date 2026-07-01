# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AMP model components including discriminator network.

This module implements the AMP-specific neural networks, particularly the
discriminator that distinguishes between agent and reference motion data.

Key Classes:
    - Discriminator: Binary classifier for agent vs. reference motions
    - AMPModel: PPO model extended with discriminator
"""

import torch
from torch import nn
from typing import List
from tensordict import TensorDict
from protomotions.utils.hydra_replacement import get_class
from protomotions.agents.ppo.model import PPOModel
from protomotions.agents.amp.config import DiscriminatorConfig, AMPModelConfig
from protomotions.agents.common.common import ModuleContainer, ObsProcessor
from protomotions.agents.common.mlp import MLPWithConcat
from protomotions.agents.base_agent.model import ProtoMotionsTensorDictModule


class Discriminator(ProtoMotionsTensorDictModule):
    """Discriminator network for AMP style rewards.

    Binary classifier that distinguishes between agent-generated and reference motion data.
    Uses ModuleContainer structure - just chains models together.

    Args:
        config: DiscriminatorConfig (extends ModuleContainerConfig).

    Attributes:
        models: ModuleContainer list of modules.
        in_keys: Input keys from config.
        out_keys: Output keys from config.
    """

    config: DiscriminatorConfig

    def __init__(self, config: DiscriminatorConfig):
        super().__init__()
        self.config = config

        # Build sequential modules
        models = []
        for input_model in config.models:
            model = get_class(input_model._target_)(config=input_model)
            models.append(model)
        self.models = nn.ModuleList(models)

        # Set TensorDict keys from config
        self.in_keys = self.config.in_keys
        self.out_keys = self.config.out_keys

        # Discover gradient penalty targets once at init
        self._grad_penalty_keys = self._find_grad_penalty_keys()
        if not self._grad_penalty_keys:
            self._grad_penalty_keys = list(self.config.in_keys)

    def _find_grad_penalty_keys(self) -> List[str]:
        """Discover tensordict keys to use as gradient penalty targets.

        Returns keys for the effective inputs to the discriminator's learned
        function — after preprocessing but before learned layers:
        - ObsProcessor outputs (normalizing or not — they preprocess raw obs)
        - MLPWithConcat internal norm_ keys (when normalize_obs=True)
        - Raw pass-through keys that bypass all preprocessing

        Some keys (e.g. masks) may not be in the autograd graph — the caller
        handles this via allow_unused=True and filters None gradients.
        """
        transformed_keys = []
        consumed_raw_keys: set = set()

        for model in self.models:
            cfg = getattr(model, "config", None)
            if cfg is None:
                continue
            if isinstance(model, ObsProcessor):
                consumed_raw_keys.update(model.in_keys)
                transformed_keys.extend(model.out_keys)
            elif isinstance(model, MLPWithConcat) and getattr(
                cfg, "normalize_obs", False
            ):
                consumed_raw_keys.update(model.in_keys)
                transformed_keys.append(f"norm_{model.config.in_keys[0]}")

        passthrough_keys = [
            k for k in self.config.in_keys if k not in consumed_raw_keys
        ]

        return transformed_keys + passthrough_keys

    def forward(
        self,
        tensordict: TensorDict,
        log_internals: bool = False,
    ) -> TensorDict:
        """Forward pass through discriminator.

        Args:
            tensordict: TensorDict containing observations.

        Returns:
            TensorDict with discriminator output added.
        """
        # Chain through all modules
        for model in self.models:
            if isinstance(model, ProtoMotionsTensorDictModule):
                tensordict = model(tensordict, log_internals=log_internals)
            else:
                tensordict = model(tensordict)
        return tensordict

    def compute_disc_reward(
        self, disc_logits: torch.Tensor, eps: float = 1e-4
    ) -> torch.Tensor:
        """Compute style reward from discriminator logits.

        Converts discriminator logits to reward using negative log probability.
        Higher reward means motion is more similar to reference data.

        Args:
            disc_logits: Discriminator logits.
            eps: Small constant for numerical stability.

        Returns:
            Style rewards for each sample (higher = more reference-like).
        """
        prob = 1 / (1 + torch.exp(-disc_logits))
        reward = -torch.log(torch.clamp(1 - prob, min=eps))
        return reward

    def all_discriminator_weights(self):
        """Get all discriminator weight matrices (works with LazyLinear).

        Returns:
            List of weight parameters from all linear layers in discriminator.
        """
        weights: List[nn.Parameter] = []
        for mod in self.modules():
            if hasattr(mod, "weight") and isinstance(mod.weight, nn.Parameter):
                weights.append(mod.weight)
        return weights

    def logit_weights(self) -> List[nn.Parameter]:
        """Get the final layer weights (logit layer).

        Returns:
            List containing the output layer weight parameter.
        """
        last_module = self.models[-1]
        if hasattr(last_module, "mlp"):
            last_module = last_module.mlp[-1]
        return [last_module.weight]


class AMPModelComponentsMixin:
    """Adds AMP discriminator modules to a host model."""

    def _build_amp_model_components(self, config):
        DiscriminatorClass = get_class(config.discriminator._target_)
        self._discriminator: Discriminator = DiscriminatorClass(
            config=config.discriminator
        )
        DiscCriticClass = get_class(config.disc_critic._target_)
        self._disc_critic: ModuleContainer = DiscCriticClass(
            config=config.disc_critic
        )
        self._validate_amp_model_keys(config)

    def _validate_amp_model_keys(self, config):
        discriminator_in_keys = list(
            set(self._discriminator.in_keys + self._disc_critic.in_keys)
        )
        discriminator_out_keys = list(
            set(self._discriminator.out_keys + self._disc_critic.out_keys)
        )
        for key in discriminator_out_keys:
            assert (
                key in config.out_keys
            ), f"Discriminator output key {key} not in out_keys {config.out_keys}"
        for key in discriminator_in_keys:
            assert (
                key in config.in_keys
            ), f"Discriminator input key {key} not in in_keys {config.in_keys}"

    def _forward_amp_model_components(
        self,
        tensordict: TensorDict,
        log_internals: bool = False,
    ) -> TensorDict:
        tensordict = self._discriminator(tensordict, log_internals=log_internals)
        tensordict = self._disc_critic(tensordict, log_internals=log_internals)
        return tensordict


class AMPModel(AMPModelComponentsMixin, PPOModel):
    """AMP model with actor, task critic, disc critic, and discriminator networks.

    Extends PPOModel by adding a discriminator network that provides style rewards
    and a separate critic for estimating discriminator reward values.

    Args:
        config: AMPModelConfig specifying all networks.

    Attributes:
        _actor: Policy network.
        _critic: Task value network.
        _disc_critic: Discriminator reward value network.
        _discriminator: Style discriminator network.
    """

    config: AMPModelConfig

    def __init__(self, config: AMPModelConfig):
        super().__init__(config)
        self._build_amp_model_components(self.config)

    def forward(
        self,
        tensordict: TensorDict,
        log_internals: bool = False,
    ) -> TensorDict:
        """Forward pass through PPO, and discriminator.

        Args:
            tensordict: TensorDict containing observations.

        Returns:
            TensorDict with all model outputs added.
        """
        tensordict = super().forward(tensordict, log_internals=log_internals)

        return self._forward_amp_model_components(
            tensordict, log_internals=log_internals
        )
