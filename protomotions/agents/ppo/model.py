# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
from protomotions.agents.common.common import ModuleContainer
from protomotions.agents.ppo.config import PPOActorConfig, PPOModelConfig
from protomotions.agents.base_agent.model import (
    BaseModel,
    ProtoMotionsTensorDictModule,
)


class PPOActor(ProtoMotionsTensorDictModule):
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
            requires_grad=self.config.learnable_std,
        )
        MuClass = get_class(self.config.mu_model._target_)
        self.mu: nn.Module = MuClass(config=self.config.mu_model)

        self.in_keys = self.config.in_keys
        self.out_keys = self.config.out_keys
        for key in ["action", "mean_action", "neglogp"]:
            assert (
                key in self.out_keys
            ), f"PPOActor output key {key} not in out_keys {self.out_keys}"

    def forward(
        self,
        tensordict: TensorDict,
        log_internals: bool = False,
    ) -> TensorDict:
        """Forward pass: compute mu/std, sample action, compute neglogp.

        Args:
            tensordict: TensorDict containing observations.

        Returns:
            TensorDict with action, mean_action, and neglogp added.
        """
        # Compute distribution parameters
        if isinstance(self.mu, ProtoMotionsTensorDictModule):
            tensordict = self.mu(tensordict, log_internals=log_internals)
        else:
            tensordict = self.mu(tensordict)
        mu = tensordict[self.config.mu_key]
        std = torch.exp(self.logstd)

        # Sample action from distribution
        # mu * 0 + std broadcasts std to match mu's batch shape while preserving
        # the computational graph when std is learnable
        dist = distributions.Normal(mu, mu * 0 + std)
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
        self._critic: ModuleContainer = CriticClass(config=self.config.critic)

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

    def forward(
        self,
        tensordict: TensorDict,
        log_internals: bool = False,
    ) -> TensorDict:
        """Forward pass through actor and critic.

        Computes all outputs:
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
        tensordict = self._actor(tensordict, log_internals=log_internals)

        # Critic forward: adds value estimate
        tensordict = self._critic(tensordict)

        return tensordict
