# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discrete autoregressive latent-prior model for supervised imitation."""

import torch
import torch.distributions as distributions
from tensordict import TensorDict
from torch import nn

from protomotions.agents.base_agent.model import BaseModel, ProtoMotionsTensorDictModule
from protomotions.agents.common.autoregressive import (
    resolve_discrete_autoregressive_config,
)
from protomotions.agents.common.discrete_latent import (
    FSQTokenization,
    make_discrete_latent_decoder,
    make_discrete_latent_target_encoder,
)
from protomotions.agents.common.latent import (
    LATENT_KEY,
    LATENT_LOGITS_KEY,
    TARGET_LATENT_KEY,
)
from protomotions.agents.common.pretrained import load_pretrained_model_module
from protomotions.utils.hydra_replacement import get_class


class DiscreteAutoregressiveLatentPriorModel(BaseModel):
    """Autoregressive discrete prior backed by a frozen tracker decoder.

    The model teacher-forces against target prior tokens during supervision
    and samples prior-token sequences during rollout/inference before unpacking
    them back to FSQ scalar codes and decoding actions.
    """

    def __init__(self, config):
        super().__init__(config)
        if (
            not config.latent_decoder.checkpoint_path
            and config.latent_decoder.module_config is None
        ):
            raise ValueError(
                "DiscreteAutoregressiveLatentPriorModelConfig.latent_decoder."
                "checkpoint_path or module_config must be set"
            )

        pretrained_module = load_pretrained_model_module(
            config.latent_decoder,
            device=torch.device("cpu"),
        )
        self.latent_decoder = make_discrete_latent_decoder(
            pretrained_module,
            freeze=config.latent_decoder.freeze,
        )
        self.latent_tokenization = FSQTokenization(
            num_fsq_levels=self.latent_decoder.num_fsq_levels,
            num_fsq_scalars=self.latent_decoder.num_fsq_scalars,
            fsq_scalars_per_prior_token=config.fsq_scalars_per_prior_token,
        )
        self.target_latent_encoder = make_discrete_latent_target_encoder(
            pretrained_module,
            tokenization=self.latent_tokenization,
            target_key=TARGET_LATENT_KEY,
            freeze=config.latent_decoder.freeze,
        )

        self.temperature = config.temperature
        self.top_p = config.top_p
        self.rollout_action_std = config.rollout_action_std

        prior_config = resolve_discrete_autoregressive_config(
            config.prior,
            num_tokens=self.latent_tokenization.num_prior_tokens,
            vocab_size=self.latent_tokenization.prior_token_vocab_size,
        )
        prior_cls = get_class(prior_config._target_)
        self.prior = prior_cls(config=prior_config)
        self.prior_context_keys = [
            key for key in self.prior.in_keys if key != self.prior.token_key
        ]
        if not self.prior_context_keys:
            raise ValueError(
                "DiscreteAutoregressiveLatentPriorModel requires the prior "
                "to declare at least one non-token input key."
            )

        self.in_keys = list(dict.fromkeys(self.get_inference_in_keys()))
        self.out_keys = ["action", "mean_action", "neglogp"]

    def compute_model_loss(
        self,
        tensordict: TensorDict,
        current_epoch: int,
        zero_loss: torch.Tensor,
        log_prefix: str = "model",
    ):
        loss = zero_loss * 0.0
        log_dict = {}
        for name, module in (
            ("latent_decoder", self.latent_decoder.decoder),
            ("target_latent_encoder", self.target_latent_encoder),
            ("prior", self.prior),
        ):
            if not isinstance(module, ProtoMotionsTensorDictModule):
                continue
            module_loss, module_log_dict = module.compute_model_loss(
                tensordict,
                current_epoch=current_epoch,
                zero_loss=zero_loss,
                log_prefix=f"{log_prefix}/{name}",
            )
            loss = loss + module_loss
            log_dict.update(module_log_dict)
        return loss, log_dict

    def optimization_module(self) -> nn.Module:
        return self.prior

    def get_inference_in_keys(self) -> list:
        decoder_in_keys = [
            key
            for key in self.latent_decoder.decoder.in_keys
            if key != self.latent_decoder.latent_key
        ]
        return list(dict.fromkeys(self.prior_context_keys + decoder_in_keys))

    def train(self, mode: bool = True):
        super().train(mode)
        self.latent_decoder.eval()
        self.target_latent_encoder.eval()
        self.prior.train(mode)
        return self

    def materialize(self, tensordict: TensorDict) -> TensorDict:
        self.target_latent_encoder(tensordict)
        prior_td = self._prior_tensordict(tensordict)
        prior_td = self.prior.generate(
            prior_td,
            num_tokens=self.latent_tokenization.num_prior_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        prior_tokens = prior_td[self.prior.generated_tokens_key]
        return self.decode_latents(tensordict, prior_tokens)

    def _decode(self, tensordict: TensorDict, fsq_codes: torch.Tensor) -> torch.Tensor:
        return self.latent_decoder.decode(tensordict, fsq_codes)

    def _prior_tensordict(
        self,
        tensordict: TensorDict,
        target_prior_tokens: torch.Tensor = None,
    ) -> TensorDict:
        data = {key: tensordict[key] for key in self.prior_context_keys}
        if target_prior_tokens is not None:
            data[self.prior.token_key] = self.latent_tokenization.one_hot_prior_tokens(
                target_prior_tokens
            )
        first_context = data[self.prior_context_keys[0]]
        return TensorDict(
            data,
            batch_size=first_context.shape[0],
            device=first_context.device,
        )

    @staticmethod
    def neglogp(actions: torch.Tensor, dist) -> torch.Tensor:
        return -dist.log_prob(actions).sum(dim=-1)

    def decode_latents(
        self,
        tensordict: TensorDict,
        prior_tokens: torch.Tensor,
    ) -> TensorDict:
        fsq_indices = self.latent_tokenization.prior_tokens_to_fsq_indices(prior_tokens)
        fsq_codes = self.latent_decoder.indices_to_codes(fsq_indices)
        action = self._decode(tensordict, fsq_codes)

        tensordict["action"] = action
        tensordict["mean_action"] = action
        tensordict[LATENT_KEY] = prior_tokens
        return tensordict

    def get_action_and_logp(
        self,
        tensordict: TensorDict,
        prior_tokens: torch.Tensor,
    ) -> TensorDict:
        tensordict = self.decode_latents(tensordict, prior_tokens)
        mean_action = tensordict["mean_action"]
        std = mean_action * 0.0 + self.rollout_action_std
        dist = distributions.Normal(mean_action, std)
        action = dist.sample()
        tensordict["action"] = action
        tensordict["neglogp"] = self.neglogp(action, dist)
        return tensordict

    def _teacher_forced(self, tensordict: TensorDict) -> TensorDict:
        target_prior_tokens = tensordict[TARGET_LATENT_KEY]
        prior_td = self._prior_tensordict(tensordict, target_prior_tokens)
        prior_td = self.prior(prior_td)
        if "action" not in tensordict.keys():
            tensordict = self.get_action_and_logp(tensordict, target_prior_tokens)
        tensordict[TARGET_LATENT_KEY] = target_prior_tokens
        tensordict[LATENT_LOGITS_KEY] = prior_td[self.prior.logits_key]
        return tensordict

    def _generate(self, tensordict: TensorDict) -> TensorDict:
        prior_td = self._prior_tensordict(tensordict)
        prior_td = self.prior.generate(
            prior_td,
            num_tokens=self.latent_tokenization.num_prior_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        prior_tokens = prior_td[self.prior.generated_tokens_key]
        logits = prior_td[self.prior.logits_key]
        tensordict = self.decode_latents(tensordict, prior_tokens)
        # This supervised model does not own a PPO action distribution on the
        # generated path. PPO-style token training must use the sampler-matched
        # token log-probs, not this action-level placeholder.
        tensordict["neglogp"] = torch.zeros(
            prior_tokens.shape[0],
            device=prior_tokens.device,
        )
        tensordict[LATENT_LOGITS_KEY] = logits
        return tensordict

    def forward(
        self,
        tensordict: TensorDict,
        log_internals: bool = False,
    ) -> TensorDict:
        if TARGET_LATENT_KEY in tensordict.keys():
            return self._teacher_forced(tensordict)
        return self._generate(tensordict)

    def collect_expert_rollout(self, tensordict: TensorDict) -> TensorDict:
        tensordict = self.target_latent_encoder(tensordict)
        return self.get_action_and_logp(tensordict, tensordict[TARGET_LATENT_KEY])

    def generate(self, tensordict: TensorDict):
        tensordict = self._generate(tensordict)
        return tensordict["action"], {
            "action": tensordict["action"],
            LATENT_KEY: tensordict[LATENT_KEY],
            LATENT_LOGITS_KEY: tensordict[LATENT_LOGITS_KEY],
        }

    def reconstruct(self, tensordict: TensorDict):
        tensordict = self.decode_latents(tensordict, tensordict[TARGET_LATENT_KEY])
        return tensordict["action"], {
            "action": tensordict["action"],
            LATENT_KEY: tensordict[LATENT_KEY],
        }

    def act(self, tensordict: TensorDict, mean: bool = True):
        return self._generate(tensordict)
