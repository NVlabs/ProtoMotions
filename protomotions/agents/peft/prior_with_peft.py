# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PEFT adapter attached to a pretrained autoregressive GPC prior.

This module owns PEFT-specific state: adapter injection, task conditioning,
and the frozen KL/sampling reference. The generic autoregressive token loop stays
on the common prior transformer.
"""

from __future__ import annotations

import copy
import logging

import torch
from torch import nn
from tensordict import TensorDict

from protomotions.agents.common.common import NormObsBase
from protomotions.agents.common.config import NormObsBaseConfig
from protomotions.agents.peft.adapters import (
    freeze_base_and_enable_peft,
    inject_transformer_peft,
    set_peft_layers_train_mode,
)

log = logging.getLogger(__name__)


class DiscretePriorWithPEFT(nn.Module):
    """Attach conditioned LoRA/DoRA layers to a frozen prior transformer.

    The wrapper consumes the prior's context keys plus one configured PEFT
    condition key. It builds ``task_c`` for the adapter layers and delegates
    token teacher-forcing/generation to ``DiscreteAutoregressiveTransformer``.
    Higher-level actors own observation preprocessing, decoding, and target
    encoding.
    """

    def __init__(
        self,
        prior,
        conditioning_dim: int | None = None,
        rank: int = 4,
        alpha: float = 0.5,
        peft_type: str = "dora",
        temperature: float = 1.0,
        top_p: float = 0.8,
        sampling_mode: str = "nucleus",
        prior_top_p: float = 0.99,
        condition_key: str = "task_cond",
        film_input_norm: bool = False,
        film_input_norm_clamp: float = 5.0,
    ):
        """Initialize PEFT wrapper.

        Args:
            prior: Base TransformerPrior model to wrap.
            conditioning_dim: Optional dimension of the full PEFT conditioning
                vector, i.e. condition_key features plus frozen-prior context
                observations. If omitted, it is inferred from warmup_obs in
                init_peft().
            rank: LoRA/DoRA rank.
            alpha: LoRA/DoRA scaling factor.
            peft_type: "lora" or "dora".
            temperature: Sampling temperature for the PEFT actor.
            top_p: Nucleus sampling threshold for the PEFT actor.
            sampling_mode: Sampling strategy - "nucleus" or "prior_constraint".
            prior_top_p: Nucleus threshold for the frozen prior (only used when
                sampling_mode="prior_constraint"). Controls how conservatively
                the frozen prior caps the PEFT actor's distribution.
            condition_key: TensorDict key carrying PEFT conditioning features.
            film_input_norm: Normalize PEFT conditioning before FiLM layers.
            film_input_norm_clamp: Clamp value for normalized conditioning.
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.peft_type = peft_type
        self.temperature = temperature
        self.top_p = top_p
        self.prior_top_p = prior_top_p
        self.condition_key = condition_key
        self.conditioning_dim = conditioning_dim
        self.sampling_mode = sampling_mode
        self.film_input_norm = (
            NormObsBase(
                NormObsBaseConfig(
                    normalize_obs=True,
                    norm_clamp_value=film_input_norm_clamp,
                )
            )
            if film_input_norm
            else None
        )

        self.base_prior = prior
        self.reference_prior = None
        self.reference_film_input_norm = None
        self._reference_ready = False

    @property
    def task_c_dim(self):
        if self.conditioning_dim is None:
            raise RuntimeError(
                "DiscretePriorWithPEFT conditioning_dim is unknown. Provide "
                "conditioning_dim or call init_peft with warmup_obs containing "
                f"{self.condition_key!r} and the frozen-prior context keys."
            )
        return self.conditioning_dim

    def init_peft(self, warmup_obs: dict | None = None):
        """Materialize prior context shape, then install PEFT adapters.

        ``warmup_obs`` should contain the raw observation keys expected by the
        frozen prior. It is only used when the pretrained prior's context
        dimension has not already been materialized by a previous forward pass.
        """
        self._prime_context_dim(warmup_obs)
        if self.conditioning_dim is None:
            if warmup_obs is None or self.condition_key not in warmup_obs:
                raise RuntimeError(
                    "DiscretePriorWithPEFT.init_peft requires warmup_obs with "
                    f"{self.condition_key!r} and frozen-prior context keys when "
                    "conditioning_dim is not provided."
                )
            self.conditioning_dim = self._build_task_c(warmup_obs, None).shape[-1]
        self.base_prior._transformer = inject_transformer_peft(
            transformer=self.base_prior._transformer,
            conditioning_dim=self.task_c_dim,
            rank=self.rank,
            alpha=self.alpha,
            peft_type=self.peft_type,
        )
        self._freeze_base()
        self._materialize_film_input_norm(warmup_obs)

    @torch.no_grad()
    def _prime_context_dim(self, warmup_obs: dict | None) -> None:
        """Run one frozen-prior teacher-forced pass if context_dim is lazy."""
        bp = self.base_prior
        if bp.context_dim is not None:
            return

        context_in_keys = list(bp.context_in_keys or [])
        if not context_in_keys:
            return
        if warmup_obs is None:
            raise RuntimeError(
                "DiscretePriorWithPEFT.init_peft requires warmup_obs when the base "
                "prior context_dim has not been materialized."
            )

        missing = [key for key in context_in_keys if key not in warmup_obs]
        if missing:
            raise RuntimeError(
                f"Prior expects context keys {context_in_keys} but warmup_obs "
                f"is missing {missing}. Available keys: {list(warmup_obs.keys())}"
            )

        device = bp._pos_emb.device
        td_data = {key: warmup_obs[key].to(device) for key in context_in_keys}
        batch_size = td_data[context_in_keys[0]].shape[0]
        if bp.token_key in warmup_obs:
            td_data[bp.token_key] = warmup_obs[bp.token_key].to(device)
        else:
            # Token values are irrelevant here; we only need a valid token tensor
            # shape so teacher_force can materialize the lazy context encoder dims.
            token_one_hot = torch.zeros(
                batch_size,
                bp.num_tokens,
                bp.vocab_size,
                device=device,
                dtype=torch.float32,
            )
            token_one_hot[:, :, 0] = 1.0
            td_data[bp.token_key] = token_one_hot
        bp.teacher_force(TensorDict(td_data, batch_size=batch_size, device=device))

    def _freeze_base(self):
        """Freeze the base prior and leave only adapter residuals trainable."""
        self.base_prior.eval()
        for p in self.base_prior.parameters():
            p.requires_grad = False
        self._freeze_base_normalizers()
        freeze_base_and_enable_peft(self.base_prior._transformer)

        if log.isEnabledFor(logging.DEBUG):
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            log.debug(
                "PEFT adapter ready: trainable %s / %s (%.2f%%), task_c_dim=%s",
                f"{trainable:,}",
                f"{total:,}",
                100 * trainable / total,
                self.task_c_dim,
            )

    @staticmethod
    def reference_state_prefixes(prefix: str = "") -> tuple[str, ...]:
        return (
            f"{prefix}reference_prior.",
            f"{prefix}reference_film_input_norm.",
        )

    def optional_full_checkpoint_state_prefixes(self) -> tuple[str, ...]:
        """State that is optional when loading full PEFT training checkpoints."""
        return ("_anchor_", "film_input_norm.", *self.reference_state_prefixes())

    @property
    def reference_ready(self) -> bool:
        return self._reference_ready

    def _freeze_base_normalizers(self) -> None:
        for module in self.base_prior.modules():
            if isinstance(module, NormObsBase):
                module._freeze_running = True

    def _freeze_normalizers(self) -> None:
        self._freeze_base_normalizers()
        if self.film_input_norm is not None:
            self.film_input_norm._freeze_running = True

    @torch.no_grad()
    def _materialize_film_input_norm(self, warmup_obs: dict | None) -> None:
        if self.film_input_norm is None or warmup_obs is None:
            return
        self._build_task_c(warmup_obs, self.film_input_norm)

    def ensure_reference_modules(self) -> None:
        if self.reference_prior is None:
            self.reference_prior = copy.deepcopy(self.base_prior)
        if self.film_input_norm is None:
            self.reference_film_input_norm = None
        elif self.reference_film_input_norm is None:
            self.reference_film_input_norm = copy.deepcopy(self.film_input_norm)
        self._freeze_reference_modules()

    def _freeze_reference_modules(self) -> None:
        if self.reference_prior is not None:
            self.reference_prior.eval()
            for parameter in self.reference_prior.parameters():
                parameter.requires_grad = False
            for module in self.reference_prior.modules():
                if isinstance(module, NormObsBase):
                    module._freeze_running = True
        if self.reference_film_input_norm is not None:
            self.reference_film_input_norm.eval()
            self.reference_film_input_norm._freeze_running = True
            for parameter in self.reference_film_input_norm.parameters():
                parameter.requires_grad = False

    def mark_reference_loaded(self) -> None:
        self.ensure_reference_modules()
        self._reference_ready = True
        log.info("Loaded PEFT prior reference from checkpoint state.")

    def require_reference(self) -> None:
        if self.reference_prior is None or not self._reference_ready:
            raise RuntimeError(
                "PEFT RLFT requires a frozen reference policy for true resume. "
                "Warm-start with load_training_state=False or use a checkpoint "
                "saved after the reference-module migration."
            )

    def capture_reference(self):
        """Pin the complete KL/sampling reference to the current PEFT policy."""
        self._freeze_normalizers()
        if self.reference_ready:
            log.debug("PEFT prior reference already pinned; leaving it unchanged.")
            return False
        self.ensure_reference_modules()
        self.reference_prior.load_state_dict(self.base_prior.state_dict())
        if self.film_input_norm is not None:
            self.reference_film_input_norm.load_state_dict(
                self.film_input_norm.state_dict()
            )
        self._freeze_reference_modules()
        self._reference_ready = True
        log.info("Pinned PEFT prior reference from current adapter state.")
        return True

    @torch.no_grad()
    def clear_peft(self):
        """Zero adapter residuals on the active student; leave the reference intact."""
        for layer in self.base_prior._transformer.layers:
            if hasattr(layer, "m"):
                layer.m.data.zero_()
            elif hasattr(layer, "lora"):
                layer.lora.B.data.zero_()

    def _build_task_c(
        self,
        input_dict: dict,
        film_input_norm: NormObsBase | None,
    ) -> torch.Tensor:
        """Return task features concatenated with raw frozen-prior context."""
        # Adapter layers see both the task command and the same raw context that
        # conditions the frozen prior. This lets PEFT modulate the prior relative
        # to the state the prior itself is using, without hard-coding terrain or
        # task-specific keys in the adapter wrapper.
        if self.condition_key not in input_dict:
            raise RuntimeError(
                f"DiscretePriorWithPEFT requires input_dict[{self.condition_key!r}]."
            )
        task_c = torch.cat(
            [input_dict[self.condition_key], self._context(input_dict)],
            dim=-1,
        )
        if film_input_norm is not None:
            task_c = film_input_norm(task_c)
        return task_c

    def _task_transformer_kwargs(self, input_dict: dict) -> dict:
        """Transformer kwargs required by PEFT-wrapped layers."""
        return {"task_c": self._build_task_c(input_dict, self.film_input_norm)}

    def _reference_task_transformer_kwargs(self, input_dict: dict) -> dict:
        """Transformer kwargs for the frozen reference policy."""
        return {
            "task_c": self._build_task_c(
                input_dict,
                self.reference_film_input_norm,
            )
        }

    def _context_items(self, input_dict: dict) -> list[tuple[str, torch.Tensor]]:
        context_in_keys = list(self.base_prior.context_in_keys)
        if not context_in_keys:
            raise RuntimeError(
                "DiscretePriorWithPEFT: base prior reports no context_in_keys; "
                "cannot route context into encode_context."
            )
        if all(key in input_dict for key in context_in_keys):
            return [(key, input_dict[key]) for key in context_in_keys]
        missing = [key for key in context_in_keys if key not in input_dict]
        raise RuntimeError(
            f"DiscretePriorWithPEFT expected frozen-prior context keys {context_in_keys}; "
            f"missing {missing}. Available keys: {list(input_dict.keys())}"
        )

    def _context(self, input_dict: dict) -> torch.Tensor:
        tensors = [tensor for _, tensor in self._context_items(input_dict)]
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim=-1)

    def _tokens(self, input_dict: dict) -> torch.Tensor:
        if self.base_prior.token_key in input_dict:
            return input_dict[self.base_prior.token_key]
        return input_dict["tokens"]

    def _context_embedding(self, input_dict: dict, prior=None) -> torch.Tensor:
        context_items = self._context_items(input_dict)
        first_context = context_items[0][1]
        td = TensorDict(
            {key: tensor for key, tensor in context_items},
            batch_size=first_context.shape[0],
            device=first_context.device,
        )
        prior = self.base_prior if prior is None else prior
        return prior.encode_context(td)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.base_prior.eval()
            if self.reference_prior is not None:
                self.reference_prior.eval()
            if self.reference_film_input_norm is not None:
                self.reference_film_input_norm.eval()
            set_peft_layers_train_mode(self.base_prior._transformer, mode)
            for layer in self.base_prior._transformer.layers:
                if hasattr(layer, "transformer_layer"):
                    layer.transformer_layer.eval()
        return self

    # ---- forward (teacher-forced) ----

    def forward(self, input_dict: dict) -> torch.Tensor:
        """Teacher-force the PEFT-wrapped prior and return token logits."""
        bp = self.base_prior
        return bp.forward_from_tokens(
            self._context_embedding(input_dict),
            self._tokens(input_dict),
            transformer_kwargs=self._task_transformer_kwargs(input_dict),
        )

    @torch.no_grad()
    def forward_prior(self, input_dict: dict) -> torch.Tensor:
        """Teacher-forced forward using frozen prior (for KL loss computation).

        Uses no_grad since prior logits are only used as targets - gradients
        flow through the PEFT model's logits only.
        """
        self.require_reference()
        reference = self.reference_prior
        return reference.forward_from_tokens(
            self._context_embedding(input_dict, prior=reference),
            self._tokens(input_dict),
            transformer_kwargs=self._reference_task_transformer_kwargs(input_dict),
        )

    # ---- generate (autoregressive) ----

    @torch.no_grad()
    def generate(
        self,
        input_dict: dict,
        return_logits: bool = True,
        return_logprob: bool = False,
    ):
        """Sample tokens from the PEFT prior, optionally constrained by reference."""
        was_training = self.training
        self.eval()
        try:
            bp = self.base_prior
            context = self._context_embedding(input_dict)
            transformer_kwargs = self._task_transformer_kwargs(input_dict)

            prior_constraint = None
            top_p = self.top_p
            if self.sampling_mode == "prior_constraint":
                # Nucleus+prior-constraint sampling first limits the candidate set by
                # the frozen reference policy, then samples with the active adapter
                # logits. The reference is fixed at RLFT start, so sampling remains
                # stable while the student adapter changes.
                if not self.reference_ready:
                    self.capture_reference()
                reference = self.reference_prior
                reference_context = self._context_embedding(input_dict, prior=reference)
                reference_kwargs = self._reference_task_transformer_kwargs(input_dict)
                top_p = self.prior_top_p

                def prior_constraint(token_indices, step):
                    return reference.next_logits_from_context(
                        reference_context,
                        token_indices=token_indices,
                        transformer_kwargs=reference_kwargs,
                    )

            indices, logits, logprob = bp.generate_from_context(
                context,
                num_tokens=bp.num_tokens,
                temperature=self.temperature,
                top_p=top_p,
                prior_constraint=prior_constraint,
                transformer_kwargs=transformer_kwargs,
            )
        finally:
            self.train(was_training)

        outputs = [indices]
        if return_logits:
            outputs.append(logits)
        if return_logprob:
            outputs.append(logprob)
        if len(outputs) > 1:
            return tuple(outputs)
        return indices
