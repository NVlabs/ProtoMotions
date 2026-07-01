# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared setup for PEFT agents built on a frozen GPC prior."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import torch
from torch import nn

from protomotions.agents.peft.utils.adapter_state import contains_adapter_state
from protomotions.agents.peft.utils.frozen_prior_checkpoint import (
    reload_frozen_base_prior_from_checkpoint,
)
from protomotions.agents.peft.utils.frozen_prior_contract import (
    resolve_frozen_prior_input_keys,
)
from protomotions.agents.peft.utils.model_state import (
    load_compatible_peft_model_state,
)
from protomotions.utils.hydra_replacement import get_class

log = logging.getLogger(__name__)


class DiscretePriorPEFTSetupMixin:
    """Load the frozen prior and build a PEFT actor around it.

    RLFT and SFT use different training loops, but they both need the same
    frozen prior dependency, adapter initialization, input validation, and slim
    adapter checkpoint handling. The loading lifecycle itself comes from
    ``PretrainedModulesMixin`` directly for SFT and through ``FineTuningAgent``
    for RLFT; this mixin only defines the PEFT-specific load contract.
    """

    require_reward_norm_on_load = False
    pretrained: Dict[str, nn.Module]

    def _pretrained_module_configs(self):
        """Return the single frozen prior dependency used to build the PEFT actor."""
        extra_modules = sorted(set(self.config.pretrained_modules) - {"prior"})
        if extra_modules:
            raise ValueError(
                "Discrete-prior PEFT pretrained_modules should contain only the "
                f"frozen 'prior' model. Unexpected modules: {extra_modules}"
            )
        prior_config = self.config.pretrained_modules.get("prior")
        if prior_config is None:
            raise RuntimeError(
                f"{type(self).__name__} requires config.pretrained_modules['prior']."
            )
        return {"prior": prior_config}

    def _pretrained_module_load_kwargs(self, _name, _pretrained_config) -> dict:
        return {"prefer_inference_config": True}

    def _post_create_model_hook(self):
        warmup_obs = self.add_agent_info_to_obs(self.env.get_obs())
        self.model._actor.init_peft(warmup_obs=warmup_obs)

    def _current_obs(self):
        with torch.no_grad():
            return self.env.get_obs()

    def _validate_peft_inputs(self):
        obs = self._current_obs()
        expected_keys = self._actor_required_input_keys()
        missing_keys = [key for key in expected_keys if key not in obs]
        if missing_keys:
            raise ValueError(
                "Discrete-prior PEFT actor inputs must be produced by the environment. "
                f"Missing keys: {missing_keys}. Available keys: {list(obs.keys())}"
            )
        return expected_keys

    def _actor_required_input_keys(self):
        actor_cfg = self.config.model.actor
        if "prior" not in self.pretrained:
            raise RuntimeError(
                f"{type(self).__name__} requires config.pretrained_modules['prior']."
            )
        keys = list(actor_cfg.in_keys)
        keys.extend(actor_cfg.peft.model.in_keys)
        keys.extend(resolve_frozen_prior_input_keys(self.pretrained["prior"]))
        return list(dict.fromkeys(keys))

    def _resolve_mimic_target_dim(self):
        obs = self._current_obs()
        return (
            obs["mimic_target_poses"].shape[-1]
            if "mimic_target_poses" in obs
            else 0
        )

    def _should_build_target_encoder(self, mimic_target_poses_dim: int) -> bool:
        # RLFT never encodes target poses into supervision tokens. SFT overrides
        # this because its expert labels come from the frozen target encoder.
        return False

    def create_model(self):
        expected_keys = self._validate_peft_inputs()
        mimic_dim = self._resolve_mimic_target_dim()
        log.info(
            "Resolved actor PEFT input keys=%s, actor.peft.model keys=%s -> %s, "
            "mimic_dim=%d",
            tuple(expected_keys),
            tuple(self.config.model.actor.peft.model.in_keys),
            tuple(self.config.model.actor.peft.model.out_keys),
            mimic_dim,
        )

        use_encoder = self._should_build_target_encoder(mimic_dim)
        ModelClass = get_class(self.config.model._target_)
        return ModelClass(
            config=self.config.model,
            pretrained_prior_model=self.pretrained["prior"],
            mimic_target_poses_dim=mimic_dim if use_encoder else 0,
        )

    def _actor_optimizer_params(self, model):
        return [p for p in model._actor.parameters() if p.requires_grad]

    def _print_param_info(self):
        actor_total = sum(p.numel() for p in self.model._actor.parameters())
        actor_trainable = sum(
            p.numel() for p in self.model._actor.parameters() if p.requires_grad
        )
        message = (
            f"PEFT parameters: actor trainable {actor_trainable:,} / "
            f"{actor_total:,} ({100 * actor_trainable / actor_total:.2f}%)"
        )
        if getattr(self, "has_critic", False):
            critic = getattr(self.model, "_critic", None)
            if critic is not None:
                critic_total = sum(p.numel() for p in critic.parameters())
                message = f"{message}; critic {critic_total:,} trainable"
        log.info(message)

    def _load_model_state_dict(self, model_state):
        load_compatible_peft_model_state(self.model, model_state)

    def _after_load_model_state_dict(self, state_dict) -> None:
        super()._after_load_model_state_dict(state_dict)
        if getattr(self, "_peft_loading_training_state", False):
            return
        self._restore_configured_frozen_prior()

    def _restore_configured_frozen_prior(self):
        """Re-apply the configured frozen prior after PEFT checkpoint loading.

        Full PEFT training checkpoints can include the frozen prior weights from
        the run that produced them. The current config is the source of truth for
        which prior this adapter should attach to, so loading finishes by
        refreshing the wrapped prior from ``pretrained_modules["prior"]``. Slim
        adapter-only inference checkpoints do not overwrite prior weights, but
        this restore is harmless and keeps full and slim checkpoints equivalent.
        """
        prior_config = self.config.pretrained_modules.get("prior")
        checkpoint_path = getattr(prior_config, "checkpoint_path", None)
        if not checkpoint_path:
            return
        reload_frozen_base_prior_from_checkpoint(
            self.model._actor.prior_with_peft,
            checkpoint_path,
            self.device,
        )

    def get_inference_state_dict(self, state_dict, model_state_dict=None):
        """Emit an adapter-only inference checkpoint."""
        adapter_state = self.model._actor.adapter_state_dict()
        state_dict.update(
            {
                "model": {
                    key: value.detach().cpu().clone()
                    for key, value in adapter_state.items()
                },
                "epoch": self.current_epoch,
                "step_count": self.step_count,
                "run_start_time": self.fit_start_time,
                "best_evaluated_score": self.best_evaluated_score,
            }
        )
        return state_dict

    def load_adapter_checkpoint(self, checkpoint: Path | str, strict: bool = True):
        """Load adapter weights over the model's existing frozen base prior."""
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        model_state = (
            state["model"] if isinstance(state, dict) and "model" in state else state
        )
        if not contains_adapter_state(model_state):
            raise RuntimeError(f"No PEFT adapter weights found in {checkpoint}")
        result = self.model._actor.load_adapter_state_dict(
            model_state,
            strict=strict,
        )
        log.info("Loaded PEFT adapter checkpoint from %s", checkpoint)
        return result
