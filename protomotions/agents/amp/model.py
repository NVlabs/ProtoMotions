# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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
from tensordict.nn import TensorDictModuleBase
from protomotions.utils.hydra_replacement import get_class
from protomotions.agents.ppo.model import PPOModel
from protomotions.agents.amp.config import DiscriminatorConfig, AMPModelConfig


class Discriminator(TensorDictModuleBase):
    """Discriminator network for AMP style rewards.

    Binary classifier that distinguishes between agent-generated and reference motion data.
    Uses SequentialModule structure - just chains modules together.

    Args:
        config: DiscriminatorConfig (extends SequentialModuleConfig).

    Attributes:
        sequential_models: Sequential list of modules.
        in_keys: Input keys from config.
        out_keys: Output keys from config.
    """

    config: DiscriminatorConfig

    def __init__(self, config: DiscriminatorConfig):
        super().__init__()
        self.config = config

        # Build sequential modules
        sequential_models = []
        for input_model in config.input_models:
            model = get_class(input_model._target_)(config=input_model)
            sequential_models.append(model)
        self.sequential_models = nn.ModuleList(sequential_models)

        # Set TensorDict keys from config
        self.in_keys = self.config.in_keys
        self.out_keys = self.config.out_keys

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Forward pass through discriminator.

        Args:
            tensordict: TensorDict containing observations.

        Returns:
            TensorDict with discriminator output added.
        """
        # Chain through all modules
        for model in self.sequential_models:
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
        weights: list[nn.Parameter] = []
        for mod in self.modules():
            if hasattr(mod, "weight") and isinstance(mod.weight, nn.Parameter):
                weights.append(mod.weight)
        return weights

    def logit_weights(self) -> List[nn.Parameter]:
        """Get the final layer weights (logit layer).

        Returns:
            List containing the output layer weight parameter.
        """
        last_module = self.sequential_models[-1]
        if hasattr(last_module, "mlp"):
            last_module = last_module.mlp[-1]
        return [last_module.weight]


class AMPModel(PPOModel):
    """AMP model with actor, critic, and discriminator networks.

    Extends PPOModel by adding a discriminator network that provides style rewards.
    The complete model includes policy, value function, and style discriminator.

    Args:
        config: AMPModelConfig specifying all three networks.

    Attributes:
        _actor: Policy network.
        _critic: Value network.
        _discriminator: Style discriminator network.
    """

    config: AMPModelConfig

    def __init__(self, config: AMPModelConfig):
        super().__init__(config)
        DiscriminatorClass = get_class(config.discriminator._target_)
        self._discriminator: Discriminator = DiscriminatorClass(
            config=self.config.discriminator
        )

        # Set in_keys from actor (actor inherits from mu model)
        discriminator_in_keys = self._discriminator.in_keys
        discriminator_out_keys = self._discriminator.out_keys
        for key in discriminator_out_keys:
            assert (
                key in self.config.out_keys
            ), f"Discriminator output key {key} not in out_keys {self.config.out_keys}"
        for key in discriminator_in_keys:
            assert (
                key in self.config.in_keys
            ), f"Discriminator input key {key} not in in_keys {self.config.in_keys}"

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Forward pass through PPO, and discriminator.

        Args:
            tensordict: TensorDict containing observations.

        Returns:
            TensorDict with all model outputs added.
        """
        tensordict = super().forward(tensordict)

        # Discriminator forward: adds discriminator output
        tensordict = self._discriminator(tensordict)
        return tensordict
