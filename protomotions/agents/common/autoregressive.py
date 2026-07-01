# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common autoregressive modules for categorical token sequences."""

from copy import deepcopy
from typing import Optional

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from protomotions.agents.base_agent.model import ProtoMotionsTensorDictModule
from protomotions.agents.common.config import (
    DiscreteAutoregressiveTransformerConfig,
    MLPWithConcatConfig,
)
from protomotions.agents.utils.training import get_activation_func
from protomotions.utils.hydra_replacement import get_class


def _validate_sampling_temperature(temperature: float) -> None:
    if temperature <= 0.0:
        raise ValueError("Sampling temperature must be positive.")


def _top_p_keep_mask(probs: torch.Tensor, p: float) -> torch.Tensor:
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cutoff = torch.cumsum(sorted_probs, dim=-1) > p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False

    keep_mask = torch.zeros_like(probs, dtype=torch.bool)
    keep_mask.scatter_(-1, sorted_idx, ~cutoff)
    return keep_mask


def _log_probs_from_filtered_probs(filtered_probs: torch.Tensor) -> torch.Tensor:
    return torch.where(
        filtered_probs > 0,
        filtered_probs.clamp(min=torch.finfo(filtered_probs.dtype).tiny).log(),
        torch.full_like(filtered_probs, -torch.inf),
    )


def _sampling_log_probs_from_keep_mask(
    logits: torch.Tensor,
    keep_mask: torch.Tensor,
    temperature: float,
    *,
    floor_on_mask: bool = False,
) -> torch.Tensor:
    probs = F.softmax(logits / temperature, dim=-1)
    filtered_probs = probs * keep_mask.float()
    filtered_probs = filtered_probs / (
        filtered_probs.sum(dim=-1, keepdim=True) + 1e-12
    )
    if floor_on_mask:
        return torch.where(
            keep_mask,
            filtered_probs.clamp(min=torch.finfo(filtered_probs.dtype).tiny).log(),
            torch.full_like(filtered_probs, -torch.inf),
        )
    return _log_probs_from_filtered_probs(filtered_probs)


def sampling_log_probs(
    logits: torch.Tensor,
    p: float = 0.9,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Log-probabilities of the actual temperature/top-p sampling distribution."""
    _validate_sampling_temperature(temperature)
    probs = F.softmax(logits / temperature, dim=-1)
    keep_mask = _top_p_keep_mask(probs, p)
    return _sampling_log_probs_from_keep_mask(logits, keep_mask, temperature)


def prior_constrained_sampling_log_probs(
    logits: torch.Tensor,
    prior_logits: torch.Tensor,
    p: float = 0.99,
    temperature: float = 1.0,
    overlap_threshold: float = 1e-3,
) -> torch.Tensor:
    """Log-probs after constraining model support to the prior top-p nucleus."""
    _validate_sampling_temperature(temperature)
    prior_probs = F.softmax(prior_logits, dim=-1)
    model_probs = F.softmax(logits / temperature, dim=-1)
    keep_mask = _top_p_keep_mask(prior_probs, p)

    filtered_probs = model_probs * keep_mask.float()
    prob_sums = filtered_probs.sum(dim=-1, keepdim=True)

    low_overlap = (prob_sums < overlap_threshold).squeeze(-1)
    if low_overlap.any():
        prior_filtered = prior_probs * keep_mask.float()
        prior_filtered = prior_filtered / (
            prior_filtered.sum(dim=-1, keepdim=True) + 1e-12
        )
        filtered_probs[low_overlap] = prior_filtered[low_overlap]
        prob_sums = filtered_probs.sum(dim=-1, keepdim=True)

    filtered_probs = filtered_probs / (prob_sums + 1e-12)
    return _log_probs_from_filtered_probs(filtered_probs)


def nucleus_sampling(logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0):
    """Sample categorical indices from the top-p nucleus."""
    probs = sampling_log_probs(logits, p=p, temperature=temperature).exp()
    return torch.multinomial(probs, 1).squeeze(-1)


def nucleus_sampling_prior_constraint(
    logits: torch.Tensor,
    prior_logits: torch.Tensor,
    p: float = 0.99,
    temperature: float = 1.0,
    overlap_threshold: float = 1e-3,
):
    """Sample from logits while restricting support to the prior top-p nucleus.

    If the model assigns effectively no probability mass to the prior nucleus,
    sample from the prior nucleus instead of falling back to unconstrained model
    probabilities. That keeps prior-constraint mode active after policy drift.
    """
    probs = prior_constrained_sampling_log_probs(
        logits,
        prior_logits,
        p=p,
        temperature=temperature,
        overlap_threshold=overlap_threshold,
    ).exp()
    return torch.multinomial(probs, 1).squeeze(-1)


def kl_divergence_categorical(
    logits: torch.Tensor,
    prior_logits: torch.Tensor,
    reduction: str = "mean",
):
    """KL divergence between categorical distributions parameterized by logits."""
    log_p = F.log_softmax(logits, dim=-1)
    log_q = F.log_softmax(prior_logits, dim=-1)
    p = F.softmax(logits, dim=-1)
    kl = (p * (log_p - log_q)).sum(dim=-1)

    if reduction == "mean":
        return kl.mean()
    if reduction == "sum":
        return kl.sum()
    return kl


def kl_divergence_from_log_probs(
    log_p: torch.Tensor,
    log_q: torch.Tensor,
    reduction: str = "mean",
):
    """KL divergence for already transformed categorical log-probabilities."""
    p = log_p.exp()
    kl_terms = torch.where(
        p > 0,
        p * (log_p - log_q),
        torch.zeros_like(p),
    )
    kl = kl_terms.sum(dim=-1)

    if reduction == "mean":
        return kl.mean()
    if reduction == "sum":
        return kl.sum()
    return kl


def kl_divergence_sampling_distribution(
    logits: torch.Tensor,
    prior_logits: torch.Tensor,
    *,
    p: float = 0.9,
    temperature: float = 1.0,
    prior_constraint: bool = False,
    reduction: str = "mean",
):
    """KL between the actual transformed token sampling distributions."""
    if prior_constraint:
        log_p = prior_constrained_sampling_log_probs(
            logits,
            prior_logits,
            p=p,
            temperature=temperature,
        )
        log_q = prior_constrained_sampling_log_probs(
            prior_logits,
            prior_logits,
            p=p,
            temperature=temperature,
        )
    else:
        _validate_sampling_temperature(temperature)
        student_probs = F.softmax(logits / temperature, dim=-1)
        student_keep_mask = _top_p_keep_mask(student_probs, p)
        prior_probs = F.softmax(prior_logits / temperature, dim=-1)
        prior_keep_mask = _top_p_keep_mask(prior_probs, p)
        reference_keep_mask = student_keep_mask | prior_keep_mask
        log_p = _sampling_log_probs_from_keep_mask(
            logits,
            student_keep_mask,
            temperature,
        )
        log_q = _sampling_log_probs_from_keep_mask(
            prior_logits,
            reference_keep_mask,
            temperature,
            floor_on_mask=True,
        )
    return kl_divergence_from_log_probs(log_p, log_q, reduction=reduction)


def generate_causal_mask(
    num_target: int,
    num_context: int = 0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build a float causal attention mask with an optional context prefix."""
    total = num_context + num_target
    mask = torch.zeros(total, total, device=device)

    if num_context > 0:
        mask[:num_context, num_context:] = float("-inf")

    if num_target > 0:
        causal = torch.triu(
            torch.ones(num_target, num_target, device=device),
            diagonal=1,
        )
        mask[num_context:, num_context:] = causal.masked_fill(
            causal == 1,
            float("-inf"),
        )

    return mask


def resolve_discrete_autoregressive_config(
    config: DiscreteAutoregressiveTransformerConfig,
    *,
    num_tokens: int,
    vocab_size: int,
) -> DiscreteAutoregressiveTransformerConfig:
    """Return a copy of ``config`` with token count and vocabulary resolved."""
    config = deepcopy(config)
    config.num_tokens = num_tokens
    config.vocab_size = vocab_size
    for model_config in config.output_head.models:
        if (
            isinstance(model_config, MLPWithConcatConfig)
            and model_config.out_keys == [config.logits_key]
        ):
            model_config.num_out = vocab_size
    return config


class DiscreteAutoregressiveTransformer(ProtoMotionsTensorDictModule):
    """Categorical autoregressive transformer with configurable projections."""

    config: DiscreteAutoregressiveTransformerConfig

    def __init__(self, config: DiscreteAutoregressiveTransformerConfig):
        ProtoMotionsTensorDictModule.__init__(self)
        self.config = config
        self.in_keys = list(config.in_keys)
        self.out_keys = list(config.out_keys)
        self.context_key = config.context_key
        self.token_key = config.token_key
        self.logits_key = config.logits_key
        self.generated_tokens_key = (
            config.generated_tokens_key or f"{config.logits_key}_tokens"
        )
        self.logprob_key = config.logprob_key
        self.context_embedding_key = config.context_embedding_key
        self.token_embedding_key = config.token_embedding_key
        self.hidden_key = config.hidden_key
        self.d_model = config.d_model
        self.num_tokens = config.num_tokens
        self.vocab_size = config.vocab_size
        self.context_in_keys = list(config.context_encoder.in_keys)
        self.context_dim = None
        if self.num_tokens <= 0 or self.vocab_size <= 1:
            raise ValueError(
                "DiscreteAutoregressiveTransformer requires resolved positive num_tokens "
                "and vocab_size > 1."
            )

        context_encoder_cls = get_class(config.context_encoder._target_)
        token_encoder_cls = get_class(config.token_encoder._target_)
        output_head_cls = get_class(config.output_head._target_)
        self._context_encoder = context_encoder_cls(config=config.context_encoder)
        self._token_encoder = token_encoder_cls(config=config.token_encoder)
        self._output_head = output_head_cls(config=config.output_head)

        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.ff_size,
            dropout=config.dropout,
            activation=get_activation_func(config.activation, return_type="functional"),
            batch_first=True,
        )
        self._transformer = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        max_seq_len = config.max_seq_len or config.num_tokens + 1
        self._pos_emb = nn.Parameter(torch.randn(1, max_seq_len, config.d_model))
        self._causal_masks = {}

    def muon_adam_fallback_modules(self):
        """Categorical input/output projections should use the auxiliary Adam path."""
        return (self._token_encoder, self._output_head)

    def compute_model_loss(
        self,
        tensordict: TensorDict,
        current_epoch: int,
        zero_loss: torch.Tensor,
        log_prefix: str = "model",
    ):
        loss = zero_loss * 0.0
        log_dict = {}
        for name, model in (
            ("context_encoder", self._context_encoder),
            ("token_encoder", self._token_encoder),
            ("output_head", self._output_head),
        ):
            if not isinstance(model, ProtoMotionsTensorDictModule):
                continue
            model_loss, model_log_dict = model.compute_model_loss(
                tensordict,
                current_epoch=current_epoch,
                zero_loss=zero_loss,
                log_prefix=f"{log_prefix}/{name}",
            )
            loss = loss + model_loss
            log_dict.update(model_log_dict)
        return loss, log_dict

    def _mask(self, num_target: int, device: torch.device):
        key = (num_target, device)
        if key not in self._causal_masks:
            self._causal_masks[key] = generate_causal_mask(
                num_target=num_target,
                num_context=1,
                device=device,
            )
        return self._causal_masks[key]

    def _one_hot_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() == 2:
            return F.one_hot(tokens.long(), self.vocab_size).float()
        if tokens.shape[-1] == self.vocab_size:
            return tokens.float()
        raise ValueError(
            f"Expected token indices (B, T) or one-hot tokens with vocab "
            f"{self.vocab_size}; got shape {tuple(tokens.shape)}"
        )

    def encode_context(self, tensordict: TensorDict) -> torch.Tensor:
        self.context_dim = sum(
            tensordict[key].shape[-1] for key in self.context_in_keys
        )
        tensordict = self._context_encoder(tensordict)
        return tensordict[self.context_embedding_key]

    def encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.shape[0]
        token_td = TensorDict(
            {self.token_key: self._one_hot_tokens(tokens)},
            batch_size=batch_size,
            device=tokens.device,
        )
        token_td = self._token_encoder(token_td)
        return token_td[self.token_embedding_key]

    def decode_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size = hidden.shape[0]
        head_td = TensorDict(
            {self.hidden_key: hidden},
            batch_size=batch_size,
            device=hidden.device,
        )
        head_td = self._output_head(head_td)
        return head_td[self.logits_key]

    def add_positions(
        self,
        sequence: torch.Tensor,
        pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        positions = self._pos_emb if pos_emb is None else pos_emb
        if sequence.shape[1] > positions.shape[1]:
            raise ValueError(
                f"Autoregressive sequence length {sequence.shape[1]} exceeds "
                f"max positional length {positions.shape[1]}"
            )
        return sequence + positions[:, : sequence.shape[1], :]

    def run_transformer(
        self,
        sequence: torch.Tensor,
        num_target: int,
        transformer: Optional[nn.Module] = None,
        transformer_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        model = self._transformer if transformer is None else transformer
        kwargs = transformer_kwargs or {}
        return model(sequence, mask=self._mask(num_target, sequence.device), **kwargs)

    def forward_from_tokens(
        self,
        context: torch.Tensor,
        tokens: torch.Tensor,
        *,
        transformer: Optional[nn.Module] = None,
        pos_emb: Optional[torch.Tensor] = None,
        transformer_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        token_emb = self.encode_tokens(tokens)
        context_emb = context.unsqueeze(1)
        sequence = torch.cat([context_emb, token_emb], dim=1)
        sequence = self.add_positions(sequence, pos_emb=pos_emb)
        hidden = self.run_transformer(
            sequence,
            num_target=token_emb.shape[1],
            transformer=transformer,
            transformer_kwargs=transformer_kwargs,
        )
        return self.decode_logits(hidden[:, : token_emb.shape[1]])

    def teacher_force(self, tensordict: TensorDict) -> TensorDict:
        context = self.encode_context(tensordict)
        logits = self.forward_from_tokens(context, tensordict[self.token_key])
        tensordict[self.logits_key] = logits
        return tensordict

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Run teacher forcing when tokens are supplied, otherwise generate tokens."""
        if self.token_key in tensordict.keys():
            return self.teacher_force(tensordict)
        return self.generate(tensordict)

    @torch.no_grad()
    def generate(
        self,
        tensordict: TensorDict,
        num_tokens: Optional[int] = None,
        *,
        temperature: float = 1.0,
        top_p: float = 0.9,
        prior_constraint=None,
    ) -> TensorDict:
        was_training = self.training
        self.eval()
        try:
            context = self.encode_context(tensordict)
            generated, logits, logps = self.generate_from_context(
                context,
                num_tokens=num_tokens or self.num_tokens,
                temperature=temperature,
                top_p=top_p,
                prior_constraint=prior_constraint,
            )
        finally:
            self.train(was_training)

        tensordict[self.generated_tokens_key] = generated
        tensordict[self.logits_key] = logits
        if self.logprob_key is not None:
            tensordict[self.logprob_key] = logps
        return tensordict

    def next_logits_from_context(
        self,
        context: torch.Tensor,
        token_indices: Optional[torch.Tensor] = None,
        *,
        transformer: Optional[nn.Module] = None,
        pos_emb: Optional[torch.Tensor] = None,
        transformer_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        """Return logits for the next token after an optional prefix."""
        context_emb = context.unsqueeze(1)
        if token_indices is None or token_indices.shape[1] == 0:
            sequence = context_emb
        else:
            token_emb = self.encode_tokens(token_indices)
            sequence = torch.cat([context_emb, token_emb], dim=1)

        sequence = self.add_positions(sequence, pos_emb=pos_emb)
        hidden = self.run_transformer(
            sequence,
            num_target=max(sequence.shape[1] - 1, 0),
            transformer=transformer,
            transformer_kwargs=transformer_kwargs,
        )
        return self.decode_logits(hidden[:, -1])

    def generate_from_context(
        self,
        context: torch.Tensor,
        num_tokens: int,
        *,
        temperature: float = 1.0,
        top_p: float = 0.9,
        prior_constraint=None,
        transformer: Optional[nn.Module] = None,
        pos_emb: Optional[torch.Tensor] = None,
        transformer_kwargs: Optional[dict] = None,
    ):
        generated = []
        all_logits = []
        all_logps = []

        for _ in range(num_tokens):
            token_indices = torch.stack(generated, dim=1) if generated else None
            step_logits = self.next_logits_from_context(
                context,
                token_indices=token_indices,
                transformer=transformer,
                pos_emb=pos_emb,
                transformer_kwargs=transformer_kwargs,
            )
            if prior_constraint is None:
                logp = sampling_log_probs(
                    step_logits,
                    p=top_p,
                    temperature=temperature,
                )
            else:
                prior_logits = prior_constraint(
                    token_indices=token_indices,
                    step=len(generated),
                )
                logp = prior_constrained_sampling_log_probs(
                    step_logits,
                    prior_logits,
                    p=top_p,
                    temperature=temperature,
                )
            next_idx = torch.multinomial(logp.exp(), 1).squeeze(-1)

            all_logps.append(logp.gather(-1, next_idx.unsqueeze(-1)).squeeze(-1))
            generated.append(next_idx)
            all_logits.append(step_logits)

        return (
            torch.stack(generated, dim=1),
            torch.stack(all_logits, dim=1),
            torch.stack(all_logps, dim=1),
        )
