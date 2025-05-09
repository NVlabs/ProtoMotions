import torch
from torch import nn
from typing import List
from hydra.utils import instantiate
from protomotions.agents.common.mlp import MLP_WithNorm
from protomotions.agents.ppo.model import PPOModel


class Discriminator(MLP_WithNorm):
    def __init__(self, config, num_in: int, num_out: int):
        super().__init__(config, num_in, num_out)

    def forward(self, input_dict: dict) -> torch.Tensor:
        outs = super().forward(input_dict)
        return torch.sigmoid(outs)

    def compute_logits(
        self, input_dict: dict, return_norm_obs: bool = False
    ) -> torch.Tensor:
        outs = super().forward(input_dict, return_norm_obs=return_norm_obs)
        return outs

    def compute_reward(self, input_dict: dict, eps: float = 1e-7) -> torch.Tensor:
        s = self.forward(input_dict)
        s = torch.clamp(s, eps, 1 - eps)
        reward = -(1 - s).log()
        return reward

    def all_discriminator_weights(self):
        weights: list[nn.Parameter] = []
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                weights.append(mod.weight)
        return weights

    def logit_weights(self) -> List[nn.Parameter]:
        return [self.mlp[-1].weight]


class AMPModel(PPOModel):
    def __init__(self, config):
        super().__init__(config)
        self._discriminator: Discriminator = instantiate(
            self.config.discriminator,
        )
