from torch import nn
import torch
from hydra.utils import instantiate
from protomotions.agents.common.mlp import MultiHeadedMLP
from protomotions.agents.common.transformer import Transformer


class DeterministicOutputModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # create networks
        self._encoder: Transformer = instantiate(
            self.config.encoder,
        )
        self._trunk: MultiHeadedMLP = instantiate(
            self.config.trunk,
        )

    def act(self, input_dict: dict):
        encoder_out = self._encoder(input_dict)
        input_dict["encoder_out"] = encoder_out
        action = self._trunk(input_dict)
        action = torch.tanh(action)

        return action
