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
from tensordict import TensorDict
from protomotions.agents.common.common import ModuleContainer
from protomotions.agents.ppo.config import PPOActorConfig, PPOModelConfig, StdMode
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
        shape = (config.num_out,)
        if config.std_mode == StdMode.PER_MOTION:
            assert config.num_motions is not None, (
                "Training setup must fill num_motions before actor construction"
            )
            if config.num_motions <= 0:
                raise ValueError(
                    "Per-motion std requires num_motions from training setup"
                )
            shape = (config.num_motions, config.num_out)
        self.logstd = nn.Parameter(
            torch.full(shape, config.actor_logstd, dtype=torch.float32),
            requires_grad=self.config.std_mode != StdMode.FIXED,
        )
        self.project_logstd()
        self.mu = self._build_submodule(self.config.mu_model)

        self.in_keys = self.config.in_keys
        self.out_keys = self.config.out_keys
        for key in ["action", "mean_action", "neglogp"]:
            assert (
                key in self.out_keys
            ), f"PPOActor output key {key} not in out_keys {self.out_keys}"

    @torch.no_grad()
    def project_logstd(self):
        """Project learned exploration after each update; fixed std remains unchanged."""
        if self.logstd.requires_grad:
            self.logstd.clamp_(self.config.min_logstd, self.config.max_logstd)

    def experience_buffer_keys(self):
        keys = super().experience_buffer_keys() + ["action_std"]
        if self.config.std_mode == StdMode.PER_MOTION:
            keys.append("global_motion_id")
        return list(dict.fromkeys(keys))

    def forward_deployable(
        self,
        tensordict: TensorDict,
        log_internals: bool = False,
    ) -> TensorDict:
        """Compute the deterministic policy without sampling exploration noise."""
        if isinstance(self.mu, ProtoMotionsTensorDictModule):
            tensordict = self.mu(tensordict, log_internals=log_internals)
        else:
            tensordict = self.mu(tensordict)
        tensordict["mean_action"] = tensordict[self.config.mu_key]
        return tensordict

    @property
    def deployable_in_keys(self) -> list:
        """The mean path reads the mu model's inputs, not the actor's full set."""
        return list(self.mu.in_keys)

    def forward(
        self,
        tensordict: TensorDict,
        log_internals: bool = False,
    ) -> TensorDict:
        """Compute the mean, sample an action and return its negative log probability."""
        tensordict = self.forward_deployable(tensordict, log_internals)
        mu = tensordict["mean_action"]
        logstd = self.logstd
        if self.config.std_mode == StdMode.PER_MOTION:
            logstd = logstd[tensordict["global_motion_id"]]
        std = logstd.exp().expand_as(mu)

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
        tensordict["action_std"] = std

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

    def experience_buffer_keys(self):
        return list(
            dict.fromkeys(
                super().experience_buffer_keys() + self._actor.experience_buffer_keys()
            )
        )

    def __init__(self, config: PPOModelConfig):
        super().__init__(config)

        # create networks
        self._actor: PPOActor = self._build_submodule(self.config.actor)
        self._critic: ModuleContainer = self._build_submodule(self.config.critic)

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

    def forward_deployable(self, tensordict: TensorDict) -> TensorDict:
        """The deployable graph is the actor's mean path; the critic is not exported."""
        return self._actor.forward_deployable(tensordict)

    @property
    def deployable_in_keys(self) -> list:
        return list(self._actor.deployable_in_keys)

    @property
    def deployable_action_key(self) -> str:
        return self._actor.deployable_action_key
