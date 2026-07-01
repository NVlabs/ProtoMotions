# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-level discrete-prior PEFT model container.

The pretrained prior is passed in by the agent. This class only wires the
PEFT actor to the optional PPO critic. Rollout is the module forward path.
"""

from __future__ import annotations

from protomotions.agents.base_agent.model import BaseModel
from tensordict import TensorDict

from protomotions.agents.peft.actor import DiscretePriorPEFTActor
from protomotions.utils.hydra_replacement import get_class


class DiscretePriorPEFTModel(BaseModel):
    """Discrete-prior PEFT actor plus optional PPO critic.

    ``forward`` is reserved for rollout TensorDicts. SFT teacher forcing lives
    on ``DiscretePriorPEFTSFTModel`` so RLFT and SFT flows stay separate.
    """

    def __init__(
        self,
        config,
        pretrained_prior_model,
        mimic_target_poses_dim: int = 0,
    ):
        super().__init__(config)
        ActorClass = get_class(config.actor._target_)
        self._actor: DiscretePriorPEFTActor = ActorClass(
            config=config.actor,
            pretrained_prior_model=pretrained_prior_model,
            mimic_target_poses_dim=mimic_target_poses_dim,
        )

        self.in_keys = list(self._actor.in_keys)
        self.out_keys = list(self._actor.out_keys)
        self._critic = None
        critic_config = getattr(config, "critic", None)
        if critic_config is not None:
            CriticClass = get_class(critic_config._target_)
            self._critic = CriticClass(config=critic_config)
            # Add critic in_keys and out_keys
            self.in_keys = list(dict.fromkeys(self.in_keys + self._critic.in_keys))
            self.out_keys = list(dict.fromkeys(self.out_keys + ["value"]))

    def optional_full_checkpoint_state_prefixes(self) -> tuple[str, ...]:
        """State sections that may differ between SFT and RLFT checkpoints."""
        return ("_critic.",)

    def _forward_critic(self, tensordict: TensorDict) -> TensorDict:
        if self._critic is None:
            return tensordict
        tensordict = self._critic(tensordict)
        out_key = self._critic.out_keys[0]
        if out_key != "value":
            tensordict["value"] = tensordict[out_key]
        return tensordict

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Run rollout inference and critic evaluation."""
        if not isinstance(tensordict, TensorDict):
            raise TypeError(
                "DiscretePriorPEFTModel.forward expects a TensorDict rollout input."
            )
        return self.forward_rollout(tensordict)

    def forward_rollout(self, tensordict: TensorDict) -> TensorDict:
        output_td = self._actor.get_action_and_logp(tensordict)
        return self._forward_critic(output_td)
