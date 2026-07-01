# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discrete autoregressive latent-prior configs for generic supervised training."""

from dataclasses import dataclass, field

from protomotions.agents.base_agent.config import (
    BaseModelConfig,
    OptimizerConfig,
)
from protomotions.agents.common.config import (
    DiscreteAutoregressiveTransformerConfig,
    PretrainedModelConfig,
)
from protomotions.agents.common.latent import (
    LATENT_KEY,
    LATENT_LOGITS_KEY,
    TARGET_LATENT_KEY,
)
from protomotions.agents.common.supervision import (
    SupervisionLossConfig,
    SupervisionLossType,
)
from protomotions.agents.supervised.config import SupervisedAgentConfig, RolloutActor


@dataclass
class DiscreteAutoregressiveLatentPriorModelConfig(BaseModelConfig):
    """Frozen latent decoder plus trainable categorical prior for GPC."""

    _target_: str = (
        "protomotions.agents.supervised.latent_prior_model."
        "DiscreteAutoregressiveLatentPriorModel"
    )

    latent_decoder: PretrainedModelConfig = field(
        default_factory=lambda: PretrainedModelConfig(
            checkpoint_path="",
            module_path="actor.mu",
        ),
        metadata={
            "help": "Frozen module exposing quantizer and decoder for generated latents."
        },
    )
    prior: DiscreteAutoregressiveTransformerConfig = field(
        default_factory=lambda: DiscreteAutoregressiveTransformerConfig(
            logits_key=LATENT_LOGITS_KEY,
            generated_tokens_key=LATENT_KEY,
        ),
        metadata={"help": "Autoregressive latent generator configuration."},
    )
    fsq_scalars_per_prior_token: int = field(
        default=1,
        metadata={
            "help": "Number of FSQ scalar indices packed into one prior token."
        },
    )
    temperature: float = field(default=1.0, metadata={"help": "Sampling temperature."})
    top_p: float = field(default=0.9, metadata={"help": "Nucleus sampling threshold."})
    rollout_action_std: float = field(
        default=0.0001,
        metadata={"help": "Gaussian action std used when collecting supervised rollouts."},
    )

    def prepare_inference_config_for_save(self) -> None:
        """Embed frozen decoder construction config for self-contained inference.

        The prior checkpoint saves the frozen latent decoder weights under this
        model. Embedding the decoder's module config into the inference resolved
        config lets `load_pretrained_model_module` rebuild that decoder even when
        the original tracker checkpoint is not available at inference time.
        """
        decoder_config = self.latent_decoder
        if decoder_config.module_config is not None:
            decoder_config.checkpoint_path = ""
            return
        if not decoder_config.checkpoint_path or not decoder_config.module_config_path:
            return

        from protomotions.agents.common.pretrained import get_path
        from protomotions.utils.config_utils import load_resolved_configs_from_checkpoint

        resolved_configs = load_resolved_configs_from_checkpoint(
            decoder_config.checkpoint_path
        )
        decoder_config.module_config = get_path(
            resolved_configs,
            decoder_config.module_config_path,
        )
        decoder_config.checkpoint_path = ""

    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(
            _target_="torch.optim.AdamW",
            lr=1e-4,
            weight_decay=0.01,
        ),
        metadata={"help": "Optimizer configuration."},
    )


@dataclass
class DiscreteAutoregressiveLatentSupervisedAgentConfig(SupervisedAgentConfig):
    """Supervised config for training a GPC autoregressive latent prior."""

    def prepare_inference_config_for_save(self) -> None:
        """Prepare nested model config before writing inference artifacts."""
        self.model.prepare_inference_config_for_save()

    model: DiscreteAutoregressiveLatentPriorModelConfig = field(
        default_factory=DiscreteAutoregressiveLatentPriorModelConfig,
        metadata={"help": "Model configuration."},
    )
    rollout_actor: RolloutActor = RolloutActor.EXPERT
    loss: SupervisionLossConfig = field(
        default_factory=lambda: SupervisionLossConfig(
            loss_type=SupervisionLossType.DISCRETE_CROSS_ENTROPY,
            prediction_key=LATENT_LOGITS_KEY,
            target_key=TARGET_LATENT_KEY,
            label_smoothing=0.001,
            log_prefix="prior",
        ),
        metadata={"help": "Autoregressive latent-token supervision loss."},
    )
