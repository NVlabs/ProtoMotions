# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MaskedMimic configs for the generic supervised agent."""

from dataclasses import dataclass, field
from enum import Enum
from protomotions.agents.base_agent.config import BaseModelConfig, OptimizerConfig
from protomotions.agents.common.config import ModuleContainerConfig
from protomotions.agents.common.supervision import SupervisionLossConfig
from protomotions.agents.supervised.config import SupervisedAgentConfig


@dataclass
class KLDScheduleConfig:
    """KL coefficient schedule for the MaskedMimic VAE posterior loss."""

    init_kld_coeff: float = field(
        default=0.0001,
        metadata={"help": "Initial KL divergence coefficient.", "min": 0.0},
    )
    end_kld_coeff: float = field(
        default=0.01,
        metadata={"help": "Final KL divergence coefficient.", "min": 0.0},
    )
    start_epoch: int = field(
        default=3000,
        metadata={"help": "Epoch to start KL coefficient annealing.", "min": 0},
    )
    end_epoch: int = field(
        default=6000,
        metadata={"help": "Epoch to end KL coefficient annealing.", "min": 0},
    )


class VAENoiseType(Enum):
    """Noise distribution used for MaskedMimic latent sampling."""

    NORMAL = "normal"
    UNIFORM = "uniform"
    ZEROS = "zeros"

    @classmethod
    def from_str(cls, value: str) -> "VAENoiseType":
        """Create enum from a case-insensitive string."""
        try:
            return next(
                member for member in cls if member.value.lower() == value.lower()
            )
        except StopIteration:
            raise ValueError(
                f"'{value}' is not a valid {cls.__name__}. "
                f"Valid values are: {[e.value for e in cls]}"
            )


@dataclass
class MaskedMimicVAEConfig:
    """VAE settings for the MaskedMimic learned-prior student."""

    kld_schedule: KLDScheduleConfig = field(
        default_factory=KLDScheduleConfig,
        metadata={"help": "KL divergence annealing schedule."},
    )
    vae_latent_dim: int = field(
        default=64,
        metadata={"help": "Dimension of VAE latent space.", "min": 1},
    )
    vae_noise_type: VAENoiseType = field(
        default=VAENoiseType.NORMAL,
        metadata={"help": "Type of latent noise: normal, uniform, or zeros."},
    )


@dataclass
class MaskedMimicModelConfig(BaseModelConfig):
    """MaskedMimic VAE learned-prior model configuration."""

    _target_: str = "protomotions.agents.supervised.masked_mimic_model.MaskedMimicModel"

    encoder: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Privileged encoder network."},
    )
    prior: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Deployable prior network."},
    )
    trunk: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Latent-to-action decoder trunk."},
    )
    vae: MaskedMimicVAEConfig = field(
        default_factory=MaskedMimicVAEConfig,
        metadata={"help": "MaskedMimic VAE settings."},
    )
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=2e-5),
        metadata={"help": "Optimizer settings for supervised training."},
    )

@dataclass
class MaskedMimicSupervisedAgentConfig(SupervisedAgentConfig):
    """MaskedMimic preset for ``SupervisedAgent``.

    The training loop is generic supervised imitation. The student is the
    MaskedMimic VAE learned-prior model, and the default supervision target is
    the privileged action produced by that model.
    """

    model: MaskedMimicModelConfig = field(
        default_factory=MaskedMimicModelConfig,
        metadata={"help": "MaskedMimic model configuration."},
    )
    loss: SupervisionLossConfig = field(
        default_factory=lambda: SupervisionLossConfig(
            prediction_key="privileged_action",
            target_key="expert_actions",
            log_prefix="masked_mimic",
        ),
        metadata={"help": "Supervision loss over MaskedMimic outputs."},
    )
