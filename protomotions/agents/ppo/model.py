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
"""PPO model implementation with actor-critic architecture.

This module implements the neural network models for Proximal Policy Optimization.
The actor outputs a Gaussian policy distribution, and the critic estimates state values.

Key Classes:
    - PPOActor: Policy network with Gaussian action distribution
    - PPOModel: Complete actor-critic model for PPO
"""

import torch
from torch import distributions, nn
from protomotions.utils.hydra_replacement import get_class
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from protomotions.agents.common.common import SequentialModuleConfig
from protomotions.agents.ppo.config import PPOActorConfig, PPOModelConfig
from protomotions.agents.base_agent.model import BaseModel


class PPOActor(TensorDictModuleBase):
    """PPO policy network (actor).

    Self-contained policy that computes distribution parameters, samples actions,
    and computes log probabilities all in a single forward pass.

    Args:
        config: Actor configuration including network architecture and initial log std.

    Attributes:
        logstd: Log standard deviation parameter (typically fixed during training).
        mu: Neural network that outputs action means.
        in_keys: List of input keys from mu model.
        out_keys: List of output keys (action, mean_action, neglogp).
    """

    def __init__(self, config: PPOActorConfig):
        super().__init__()
        self.config = config
        self.logstd = nn.Parameter(
            torch.ones(self.config.num_out) * self.config.actor_logstd,
            requires_grad=False,
        )
        MuClass = get_class(self.config.mu_model._target_)
        self.mu: TensorDictModuleBase = MuClass(config=self.config.mu_model)

        self.in_keys = self.config.in_keys
        self.out_keys = self.config.out_keys
        for key in ["action", "mean_action", "neglogp"]:
            assert (
                key in self.out_keys
            ), f"PPOActor output key {key} not in out_keys {self.out_keys}"

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Forward pass: compute mu/std, sample action, compute neglogp.

        This is the only method - self-contained and clean.

        Args:
            tensordict: TensorDict containing observations.

        Returns:
            TensorDict with action, mean_action, and neglogp added.
        """
        # Compute distribution parameters
        tensordict = self.mu(tensordict)
        mu = tensordict[self.config.mu_key]
        std = torch.exp(self.logstd)

        # Sample action from distribution
        dist = distributions.Normal(mu, std)
        action = dist.sample()

        # Compute negative log probability
        neglogp = -dist.log_prob(action).sum(dim=-1)

        # Store all outputs
        tensordict["action"] = action
        tensordict["mean_action"] = mu
        tensordict["neglogp"] = neglogp

        return tensordict


class PPOModel(BaseModel):
    """Complete PPO model with actor and critic networks.

    Pure forward function that computes all model outputs in TensorDict.
    The forward pass adds action distribution parameters and value estimates.

    Args:
        config: Model configuration specifying actor and critic architectures.

    Attributes:
        _actor: Policy network.
        _critic: Value network.
    """

    config: PPOModelConfig

    def __init__(self, config: PPOModelConfig):
        super().__init__(config)

        # create networks
        ActorClass = get_class(self.config.actor._target_)
        self._actor: PPOActor = ActorClass(config=self.config.actor)

        CriticClass = get_class(self.config.critic._target_)
        self._critic: SequentialModuleConfig = CriticClass(config=self.config.critic)

        # Set in_keys from actor (actor inherits from mu model)
        actor_critic_in_keys = list(set(self._actor.in_keys + self._critic.in_keys))
        actor_critic_out_keys = list(set(self._actor.out_keys + self._critic.out_keys))
        for key in actor_critic_out_keys:
            assert (
                key in self.config.out_keys
            ), f"PPOModel output key {key} not in out_keys {self.config.out_keys}"
        for key in actor_critic_in_keys:
            assert (
                key in self.config.in_keys
            ), f"PPOModel input key {key} not in in_keys {self.config.in_keys}"

        self.in_keys = self.config.in_keys
        self.out_keys = self.config.out_keys

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Forward pass through actor and critic.

        This is the main interface for the model. Computes all outputs:
        - action: Sampled action
        - mean_action: Deterministic action (mean)
        - neglogp: Negative log probability of sampled action
        - value: State value estimate

        Args:
            tensordict: TensorDict containing observations.

        Returns:
            TensorDict with all model outputs added.
        """
        # Actor forward: adds action, mean_action, neglogp
        tensordict = self._actor(tensordict)

        # Critic forward: adds value estimate
        tensordict = self._critic(tensordict)

        return tensordict
