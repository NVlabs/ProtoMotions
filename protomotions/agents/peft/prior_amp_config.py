# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for discrete-prior PEFT with AMP style rewards."""

from dataclasses import dataclass, field
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from protomotions.envs.mdp_component import MdpComponent

from protomotions.agents.amp.config import AMPParametersConfig, DiscriminatorConfig
from protomotions.agents.common.config import ModuleContainerConfig
from protomotions.agents.base_agent.config import OptimizerConfig
from protomotions.agents.peft.prior_config import (
    DiscretePriorPEFTRLFTAgentConfig,
    DiscretePriorPEFTRLFTModelConfig,
)


@dataclass
class DiscretePriorPEFTRLFTAMPModelConfig(DiscretePriorPEFTRLFTModelConfig):
    """Discrete-prior PEFT RLFT model plus AMP discriminator modules."""

    _target_: str = (
        "protomotions.agents.peft.prior_amp_model."
        "DiscretePriorPEFTRLFTAMPModel"
    )
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    discriminator_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=1e-4),
    )
    disc_critic: ModuleContainerConfig = field(default_factory=ModuleContainerConfig)
    disc_critic_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=1e-4),
    )


@dataclass
class DiscretePriorPEFTRLFTAMPAgentConfig(DiscretePriorPEFTRLFTAgentConfig):
    """Discrete-prior PEFT RLFT agent augmented with AMP rewards."""

    _target_: str = (
        "protomotions.agents.peft.prior_amp_agent."
        "DiscretePriorPEFTRLFTAMPAgent"
    )
    model: DiscretePriorPEFTRLFTAMPModelConfig = field(
        default_factory=DiscretePriorPEFTRLFTAMPModelConfig
    )
    amp_parameters: AMPParametersConfig = field(default_factory=AMPParametersConfig)
    reference_obs_components: Dict[str, "MdpComponent"] = field(default_factory=dict)
