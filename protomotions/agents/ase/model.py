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
import torch
from torch import nn
from typing import List
from tensordict import TensorDict
from protomotions.agents.amp.model import Discriminator, AMPModel
from protomotions.agents.ase.config import ASEDiscriminatorEncoderConfig, ASEModelConfig
from protomotions.agents.common.common import ModuleContainer


DISC_LOGIT_INIT_SCALE = 1.0
ENC_LOGIT_INIT_SCALE = 0.1


class ASEDiscriminatorEncoder(Discriminator):
    """Discriminator with MI encoder head for ASE.

    Inherits from Discriminator and adds an encoder head for mutual information learning.
    """

    config: ASEDiscriminatorEncoderConfig

    def __init__(self, config: ASEDiscriminatorEncoderConfig):
        super().__init__(config)

        self._encoder_initialized = False

    def _initialize_encoder_weights(self):
        """Initialize encoder weights after materialization."""
        encoder = None
        final_module = self.models[-1]
        for model in final_module.output_models:
            if model.out_keys[0] == "mi_enc_output":  # Found the encoder module
                encoder = model
                break

        assert encoder is not None, "Encoder module not found"

        if (
            not self._encoder_initialized
            and hasattr(encoder, "weight")
            and encoder.weight is not None
        ):
            torch.nn.init.uniform_(
                encoder.weight, -ENC_LOGIT_INIT_SCALE, ENC_LOGIT_INIT_SCALE
            )
            torch.nn.init.zeros_(encoder.bias)
            self._encoder_initialized = True

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Forward pass computing discriminator and MI encoder outputs.

        Args:
            tensordict: TensorDict containing observations and latents.

        Returns:
            TensorDict with disc_logits and mi_enc_output added.
        """
        # Call parent (Discriminator forward) - adds disc_logits
        tensordict = super().forward(tensordict)

        # Initialize encoder weights after materialization
        self._initialize_encoder_weights()

        return tensordict

    def compute_mi_reward(
        self, tensordict: TensorDict, mi_hypersphere_reward_shift: bool
    ):
        """Computes the Mutual Information based reward.

        Args:
            tensordict: TensorDict with mi_enc_output and latents.
            mi_hypersphere_reward_shift: Whether to shift reward to [0, 1].

        Returns:
            torch.Tensor: Mutual Information reward tensor.
        """
        enc_pred = tensordict["mi_enc_output"]
        latents = tensordict["latents"]
        neg_err = -self.calc_von_mises_fisher_enc_error(enc_pred, latents)
        if mi_hypersphere_reward_shift:
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

    def _get_weights_from_module(self, module):
        """Helper to recursively get weights by explicitly traversing structure.

        Args:
            module: Module to extract weights from.

        Returns:
            List of weight parameters.
        """
        weights = []

        # If it's a ModuleContainer, recursively process its models
        if hasattr(module, "models"):
            for sub_model in module.models:
                weights.extend(self._get_weights_from_module(sub_model))

        # If it has an mlp ModuleContainer, process that
        elif hasattr(module, "mlp") and isinstance(module.mlp, nn.Sequential):
            for layer in module.mlp:
                if hasattr(layer, "weight") and isinstance(layer.weight, nn.Parameter):
                    weights.append(layer.weight)

        # Otherwise, check if this module itself has weights
        elif hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
            weights.append(module.weight)

        return weights

    def all_weights(self):
        """Returns all weights from all sequential modules (trunk + discriminator + encoder).

        Uses explicit walking to avoid duplicates in nested structures.

        Returns:
            List[nn.Parameter]: List of all weight parameters.
        """
        weights: List[nn.Parameter] = []

        # Walk through all sequential models
        for model in self.models:
            if isinstance(model, ModuleContainer):
                # Include all output models (both discriminator and encoder)
                for output_model in model.models:
                    weights.extend(self._get_weights_from_module(output_model))
            else:
                weights.extend(self._get_weights_from_module(model))

        return weights

    def all_discriminator_weights(self):
        """Returns weights of discriminator part only (excludes encoder head).

        Explicitly walks through sequential_models to avoid including encoder head.

        Returns:
            List[nn.Parameter]: List of discriminator weight parameters.
        """
        weights: List[nn.Parameter] = []

        # Walk through sequential models
        for model in self.models:
            if isinstance(model, ModuleContainer):
                # Only include discriminator head, not encoder
                for output_model in model.models:
                    if (
                        hasattr(output_model, "out_keys")
                        and "mi_enc_output" in output_model.out_keys
                    ):
                        continue  # Skip encoder head
                    # Include this output module's weights
                    weights.extend(self._get_weights_from_module(output_model))
            else:
                # Include all weights from this module
                weights.extend(self._get_weights_from_module(model))

        return weights

    def logit_weights(self) -> List[nn.Parameter]:
        """Returns the weights of the final discriminator layer.

        Returns:
            List[nn.Parameter]: List containing the weight parameter of the discriminator's output layer.
        """
        weights = []

        # Find discriminator head in ModuleContainer
        for model in self.models:
            if isinstance(model, ModuleContainer):
                for output_model in model.models:
                    # Find the discriminator head (outputs disc_logits)
                    if (
                        hasattr(output_model, "out_keys")
                        and "disc_logits" in output_model.out_keys
                    ):
                        # Get the final layer weight
                        if hasattr(output_model, "mlp") and len(output_model.mlp) > 0:
                            final_layer = output_model.mlp[-1]
                            if hasattr(final_layer, "weight"):
                                weights.append(final_layer.weight)
                        break
                break

        return weights

    def all_enc_weights(self):
        """Returns all weights of the encoder part only (includes trunk + encoder head).

        Returns:
            List[nn.Parameter]: List of encoder weight parameters.
        """
        weights: List[nn.Parameter] = []

        # Get trunk weights (all Sequential modules before MultiOutput)
        for model in self.models:
            if isinstance(model, ModuleContainer):
                # Found MultiOutput, only get encoder head weights
                for output_model in model.models:
                    if (
                        hasattr(output_model, "out_keys")
                        and "mi_enc_output" in output_model.out_keys
                    ):
                        # This is the encoder head
                        weights.extend(self._get_weights_from_module(output_model))
                break  # Don't continue past MultiOutput
            else:
                # Include trunk weights
                weights.extend(self._get_weights_from_module(model))

        return weights

    def enc_weights(self) -> List[nn.Parameter]:
        """Returns the weights of the final encoder layer only.

        Returns:
            List[nn.Parameter]: List containing the weight parameter of the encoder's output layer.
        """
        weights = []
        # Find the encoder head in MultiOutputModule
        for model in self.models:
            if isinstance(model, ModuleContainer):
                for output_model in model.models:
                    if (
                        hasattr(output_model, "out_keys")
                        and "mi_enc_output" in output_model.out_keys
                    ):
                        # Get the final layer weight
                        if hasattr(output_model, "mlp") and len(output_model.mlp) > 0:
                            final_layer = output_model.mlp[-1]
                            if hasattr(final_layer, "weight"):
                                weights.append(final_layer.weight)
                        break
                break
        return weights


class ASEModel(AMPModel):
    """ASE model with actor, task critic, disc critic, MI critic, and discriminator.

    Extends AMPModel by adding an MI critic for estimating MI reward values.

    Args:
        config: ASEModelConfig specifying all networks.

    Attributes:
        _actor: Policy network.
        _critic: Task value network.
        _disc_critic: Discriminator reward value network.
        _mi_critic: MI reward value network.
        _discriminator: Style discriminator with MI encoder.
    """

    config: ASEModelConfig

    def __init__(self, config: ASEModelConfig):
        super().__init__(config)
        self._mi_critic = ModuleContainer(config=self.config.mi_critic)

        mi_in_keys = list(set(self._mi_critic.in_keys))
        mi_out_keys = list(set(self._mi_critic.out_keys))
        for key in mi_out_keys:
            assert (
                key in self.config.out_keys
            ), f"MI critic output key {key} not in out_keys {self.config.out_keys}"
        for key in mi_in_keys:
            assert (
                key in self.config.in_keys
            ), f"MI critic input key {key} not in in_keys {self.config.in_keys}"

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Forward pass through AMP model and MI critic.

        Args:
            tensordict: TensorDict containing observations.

        Returns:
            TensorDict with all model outputs added.
        """
        tensordict = super().forward(tensordict)
        tensordict = self._mi_critic(tensordict)
        return tensordict
