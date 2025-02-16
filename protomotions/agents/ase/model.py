# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
from torch import nn
from typing import List
from hydra.utils import instantiate
from protomotions.agents.common.mlp import MLP_WithNorm
from protomotions.agents.ppo.model import PPOModel
from protomotions.utils import model_utils

DISC_LOGIT_INIT_SCALE = 1.0
ENC_LOGIT_INIT_SCALE = 0.1


class ASEDiscriminatorEncoder(nn.Module):
    def __init__(self, config, num_in: int):
        super().__init__()
        self.config = config
        self.trunk = MLP_WithNorm(config.trunk, num_in, config.trunk.num_out)
        self.trunk_output_activation = model_utils.get_activation_func(config.trunk.output_activation)
        
        self.encoder = torch.nn.Linear(config.trunk.num_out, config.encoder.num_out)
        torch.nn.init.uniform_(self.encoder.weight, -ENC_LOGIT_INIT_SCALE, ENC_LOGIT_INIT_SCALE)
        torch.nn.init.zeros_(self.encoder.bias)
        
        self.discriminator = torch.nn.Linear(config.trunk.num_out, 1)
        torch.nn.init.uniform_(self.discriminator.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)

    def forward(self, input_dict: dict) -> torch.Tensor:
        """Forward pass through the discriminator network.

        Args:
            input_dict (dict): Dictionary containing input tensors, must contain 'historical_self_obs'.

        Returns:
            torch.Tensor: Discriminator output, sigmoid probability of being expert data. Shape (batch_size, 1).
        """
        outs = self.trunk(input_dict)
        outs = self.trunk_output_activation(outs)
        outs = self.discriminator(outs)
        return torch.sigmoid(outs)
    
    def mi_enc_forward(self, input_dict: dict) -> torch.Tensor:
        """Forward pass through the encoder network for Mutual Information maximization.

        Args:
            input_dict (dict): Dictionary containing input tensors, must contain 'historical_self_obs'.

        Returns:
            torch.Tensor: Normalized encoder output. Shape (batch_size, encoder_output_dim).
        """
        outs = self.trunk(input_dict)
        outs = self.trunk_output_activation(outs)
        outs = self.encoder(outs)
        return torch.nn.functional.normalize(outs, dim=-1)

    def compute_logits(
        self, input_dict: dict, return_norm_obs: bool = False
    ) -> torch.Tensor:
        """Computes logits from the discriminator network.

        Args:
            input_dict (dict): Dictionary containing input tensors, must contain 'historical_self_obs'.
            return_norm_obs (bool, optional): Whether to return normalized observations. Defaults to False.

        Returns:
            torch.Tensor: Discriminator logits. Shape (batch_size, 1) or dictionary with logits and normalized observations.
        """
        outs = self.trunk(input_dict, return_norm_obs=return_norm_obs)
        if return_norm_obs:
            outs["outs"] = self.trunk_output_activation(outs["outs"])
            outs["outs"] = self.discriminator(outs["outs"])
        else:
            outs = self.trunk_output_activation(outs)
            outs = self.discriminator(outs)
        
        return outs

    def compute_reward(self, input_dict: dict, eps: float = 1e-7) -> torch.Tensor:
        """Computes the reward signal from the discriminator output.

        Args:
            input_dict (dict): Dictionary containing input tensors, must contain 'historical_self_obs'.
            eps (float, optional): Small epsilon value to clamp discriminator output for numerical stability. Defaults to 1e-7.

        Returns:
            torch.Tensor: Reward tensor. Shape (batch_size, 1).
        """
        s = self.forward(input_dict)
        s = torch.clamp(s, eps, 1 - eps)
        reward = -(1 - s).log()
        return reward

    def compute_mi_reward(self, obs, latents):
        """Computes the Mutual Information based reward.

        Args:
            obs (torch.Tensor): Observation tensor. Shape (batch_size, obs_dim).
            latents (torch.Tensor): Latent variable tensor. Shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Mutual Information reward tensor. Shape (batch_size, 1).
        """
        obs = self.trunk(obs)
        obs = self.trunk_output_activation(obs)
        enc_pred = self.encoder(obs)
        enc_pred = torch.nn.functional.normalize(enc_pred, dim=-1)
        
        neg_err = -self.calc_von_mises_fisher_enc_error(enc_pred, latents)
        if self.config.ase_parameters.mi_hypersphere_reward_shift:
            reward = (neg_err + 1) / 2
        else:
            reward = torch.clamp_min(neg_err, 0.0)
        
        return reward
    
    def calc_von_mises_fisher_enc_error(self, enc_pred, latent):
        """Calculates the Von Mises-Fisher error between predicted and true latent vectors.

        Args:
            enc_pred (torch.Tensor): Predicted encoded latent vector. Shape (batch_size, latent_dim).
            latent (torch.Tensor): True latent vector. Shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Von Mises-Fisher error. Shape (batch_size, 1).
        """
        err = enc_pred * latent
        err = -torch.sum(err, dim=-1, keepdim=True)
        return err

    def all_weights(self):
        """Returns all weights of the discriminator encoder network.

        Returns:
            List[nn.Parameter]: List of weight parameters.
        """
        weights: list[nn.Parameter] = []
        for mod in self.trunk.mlp.modules():
            if isinstance(mod, nn.Linear):
                weights.append(mod.weight)
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                weights.append(mod.weight)
        return weights

    def all_discriminator_weights(self):
        """Returns all weights of the discriminator part of the network.

        Returns:
            List[nn.Parameter]: List of discriminator weight parameters.
        """
        weights: list[nn.Parameter] = []
        for mod in self.trunk.mlp.modules():
            if isinstance(mod, nn.Linear):
                weights.append(mod.weight)
        weights.append(self.logit_weights()[0])
        return weights

    def logit_weights(self) -> List[nn.Parameter]:
        """Returns the weights of the final discriminator layer.

        Returns:
            List[nn.Parameter]: List containing the weight parameter of the discriminator's linear layer.
        """
        return [self.discriminator.weight]
    
    def all_enc_weights(self):
        """Returns all weights of the encoder part of the network.

        Returns:
            List[nn.Parameter]: List of encoder weight parameters.
        """
        weights: list[nn.Parameter] = []
        for mod in self.trunk.mlp.modules():
            if isinstance(mod, nn.Linear):
                weights.append(mod.weight)
        weights.append(self.encoder.weight)
        return weights

    def enc_weights(self) -> List[nn.Parameter]:
        """Returns the weights of the final encoder layer.

        Returns:
            List[nn.Parameter]: List containing the weight parameter of the encoder's linear layer.
        """
        return [self.encoder.weight]


class ASEModel(PPOModel):
    def __init__(self, config):
        super().__init__(config)
        self._discriminator: ASEDiscriminatorEncoder = instantiate(
            self.config.ase_discriminator_encoder,
        )
