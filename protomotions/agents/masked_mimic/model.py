import torch
from torch import nn
from hydra.utils import instantiate
from protomotions.agents.common.mlp import MultiHeadedMLP


class VaeModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self._trunk = instantiate(self.config.trunk)
        self._mu_head = instantiate(self.config.mu_head)
        self._logvar_head = instantiate(self.config.logvar_head)

    def forward(self, input_dict):
        trunk_out = self._trunk(input_dict)
        mu = self._mu_head(trunk_out)
        logvar = self._logvar_head(trunk_out)
        return {"mu": mu, "logvar": logvar}


class VaeDeterministicOutputModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # create networks
        self._encoder: VaeModule = instantiate(
            self.config.encoder,
        )
        self._prior: VaeModule = instantiate(
            self.config.prior,
        )
        self._trunk: MultiHeadedMLP = instantiate(
            self.config.trunk,
        )

    def reparameterization(self, mean, std, vae_noise):
        z = mean + std * vae_noise  # reparameterization trick
        return z

    def act(self, input_dict: dict, with_encoder: bool = False):
        prior_out = self._prior(input_dict)
        if with_encoder:
            encoder_out = self._encoder(input_dict)
            mu = prior_out["mu"] + encoder_out["mu"]
            logvar = encoder_out["logvar"]
        else:
            mu = prior_out["mu"]
            logvar = prior_out["logvar"]

        z = self.reparameterization(
            mu,
            torch.exp(0.5 * logvar),
            input_dict["vae_noise"],
        )
        input_dict["vae_latent"] = z
        action = self._trunk(input_dict)
        action = torch.tanh(action)
        
        return action

    def get_action_and_vae_outputs(self, input_dict: dict):
        """Get action and VAE outputs by sampling from the encoder.

        This method is used during training to sample from both the encoder and prior networks.
        The encoder output's mu acts as a residual to the prior's mu, while its logvar is used directly.
        This allows the encoder to refine the prior's prediction while maintaining the prior's
        influence on the latent space.

        Args:
            input_dict: Dictionary containing model inputs

        Returns:
            Tuple containing:
            - action: The output action after passing through trunk network
            - prior_out: Output dictionary from prior network with mu/logvar
            - encoder_out: Output dictionary from encoder network with mu (as residual) and logvar
        """
        prior_out = self._prior(input_dict)
        encoder_out = self._encoder(input_dict)

        mu = prior_out["mu"] + encoder_out["mu"]
        logvar = encoder_out["logvar"]

        if "vae_noise" not in input_dict:
            # During training, we randomly re-sample the noise.
            input_dict["vae_noise"] = torch.randn_like(mu)

        z = self.reparameterization(
            mu,
            torch.exp(0.5 * logvar),
            input_dict["vae_noise"],
        )

        input_dict["vae_latent"] = z
        action = self._trunk(input_dict)
        action = torch.tanh(action)

        return action, prior_out, encoder_out

    @staticmethod
    def kl_loss(prior_outs, encoder_outs):
        return 0.5 * (
            prior_outs["logvar"]
            - encoder_outs["logvar"]
            + torch.exp(encoder_outs["logvar"]) / torch.exp(prior_outs["logvar"])
            + encoder_outs["mu"] ** 2 / torch.exp(prior_outs["logvar"])
            - 1
        )
