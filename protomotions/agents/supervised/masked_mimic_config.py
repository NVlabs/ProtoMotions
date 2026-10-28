# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MaskedMimic configs for the generic supervised agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import List, Optional
from protomotions.agents.base_agent.config import BaseModelConfig, OptimizerConfig
from protomotions.agents.common.config import ModuleContainerConfig
from protomotions.agents.common.supervision import SupervisionLossConfig
from protomotions.agents.supervised.config import SupervisedAgentConfig
from protomotions.agents.evaluators.config import MaskedMimicActionSource


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
    cyclical: bool = field(
        default=False,
        metadata={
            "help": "Repeat the inclusive start/end ramp after end_epoch, "
            "restarting at start_epoch on end_epoch + 1."
        },
    )

    def __post_init__(self):
        if self.cyclical and self.end_epoch <= self.start_epoch:
            raise ValueError("Cyclical KLD requires end_epoch > start_epoch")


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


class LatentGeometry(Enum):
    """Geometry of samples passed to the VAE decoder."""

    GAUSSIAN = "gaussian"
    SPHERE = "sphere"


@dataclass
class MaskedMimicVAEConfig:
    """VAE settings for the MaskedMimic learned-prior student."""

    kld_schedule: Optional[KLDScheduleConfig] = field(
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
    latent_geometry: LatentGeometry = field(
        default=LatentGeometry.GAUSSIAN,
        metadata={"help": "Decoder latent geometry: gaussian or sphere."},
    )
    prior_mean_unit_norm_coeff: float = field(
        default=0.0,
        metadata={
            "help": "Penalty on the learned prior mean's distance from unit norm."
        },
    )

    def __post_init__(self):
        if not isinstance(self.latent_geometry, LatentGeometry):
            raise ValueError("latent_geometry must be a LatentGeometry enum member")
        if self.vae_latent_dim < 1:
            raise ValueError("vae_latent_dim must be positive")
        if (
            not math.isfinite(self.prior_mean_unit_norm_coeff)
            or self.prior_mean_unit_norm_coeff < 0
        ):
            raise ValueError(
                "prior_mean_unit_norm_coeff must be finite and nonnegative"
            )
        if self.prior_mean_unit_norm_coeff and self.latent_geometry != LatentGeometry.SPHERE:
            raise ValueError("The unit-norm penalty requires sphere latent geometry")


@dataclass
class TemporalLossConfig:
    """Stop-gradient MSE between a model output and an adjacent rollout value.

    ``prediction_key`` (default ``key``) is compared against ``key`` from the
    previous step, or from the next step when ``forward`` is set. Both keys
    must be declared model outputs. Pairs that cross an episode reset are
    excluded.
    """

    key: str
    coefficient: float
    prediction_key: Optional[str] = None
    forward: bool = False

    def __post_init__(self):
        if self.prediction_key is None:
            self.prediction_key = self.key
        if (
            not self.key
            or not self.prediction_key
            or not math.isfinite(self.coefficient)
            or self.coefficient < 0
        ):
            raise ValueError(
                "Temporal loss requires keys and a finite nonnegative coefficient"
            )

    @property
    def target_key(self) -> str:
        return f"{self.direction}_target_{self.key}"

    @property
    def valid_key(self) -> str:
        return f"{self.direction}_target_valid"

    @property
    def direction(self) -> str:
        return "next" if self.forward else "previous"


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
    temporal_losses: List[TemporalLossConfig] = field(
        default_factory=list,
        metadata={"help": "Stop-gradient MSE terms against adjacent rollout outputs."},
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

    def apply_inference_defaults(self, env) -> None:
        """Student alone (the supervised cleanup), driven by the learned prior.

        Training evaluates the posterior (it has the full future); inference has
        only the masked goal, so the prior is the action path.
        """
        super().apply_inference_defaults(env)
        self.evaluator.action_source = MaskedMimicActionSource.PRIOR
