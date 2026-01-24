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
"""Configuration classes for ASE (Adversarial Skill Embeddings) agent.

ASE extends AMP with a learned latent skill space that enables diverse behaviors
and mutual information maximization between latent codes and behaviors.
"""

from typing import List
from dataclasses import dataclass, field
from protomotions.agents.amp.config import AMPAgentConfig, AMPModelConfig, DiscriminatorConfig
from protomotions.agents.common.config import ModuleContainerConfig
from protomotions.agents.ppo.config import OptimizerConfig


@dataclass
class ASEParametersConfig:
    """Configuration for ASE-specific hyperparameters."""

    latent_dim: int = field(
        default=64,
        metadata={"help": "Dimension of the latent skill space.", "min": 1}
    )
    latent_steps_min: int = field(
        default=1,
        metadata={"help": "Minimum steps before resampling latent.", "min": 1}
    )
    latent_steps_max: int = field(
        default=150,
        metadata={"help": "Maximum steps before resampling latent.", "min": 1}
    )

    mi_reward_w: float = field(
        default=0.5,
        metadata={"help": "Weight for mutual information reward.", "min": 0.0}
    )
    mi_hypersphere_reward_shift: bool = field(
        default=True,
        metadata={"help": "Shift MI reward to encourage hypersphere projections."}
    )

    mi_enc_weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay for MI encoder parameters.", "min": 0.0}
    )
    mi_enc_grad_penalty: float = field(
        default=0.0,
        metadata={"help": "Gradient penalty for MI encoder.", "min": 0.0}
    )

    diversity_bonus: float = field(
        default=0.01,
        metadata={"help": "Bonus reward for behavior diversity.", "min": 0.0}
    )
    diversity_tar: float = field(
        default=1.0,
        metadata={"help": "Target diversity level.", "min": 0.0}
    )

    latent_uniformity_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for latent uniformity loss.", "min": 0.0}
    )
    uniformity_kernel_scale: float = field(
        default=1.0,
        metadata={"help": "Scale for uniformity kernel.", "min": 0.0}
    )


@dataclass
class ASEDiscriminatorEncoderConfig(DiscriminatorConfig):
    """Configuration for ASE Discriminator-Encoder network (extends ModuleContainerConfig)."""

    encoder_out_size: int = field(
        default=None,
        metadata={"help": "Output size for encoder. Should match latent_dim.", "min": 1}
    )
    _target_: str = "protomotions.agents.ase.model.ASEDiscriminatorEncoder"
    in_keys: List[str] = field(
        default_factory=list,
        metadata={"help": "Input observation keys."}
    )
    out_keys: List[str] = field(
        default_factory=lambda: ["disc_logits", "mi_enc_output"],
        metadata={"help": "Output keys for discriminator logits and MI encoder output."}
    )

    def __post_init__(self):
        getattr(
            super(), "__post_init__", lambda: None
        )()  # Call super's __post_init__ if it exists
        assert self.encoder_out_size is not None, "encoder_out_size must be provided"


@dataclass
class ASEModelConfig(AMPModelConfig):
    """Configuration for ASE model with MI critic."""

    _target_: str = "protomotions.agents.ase.model.ASEModel"
    mi_critic: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Critic network for mutual information reward."}
    )
    mi_critic_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=1e-4),
        metadata={"help": "Optimizer settings for MI critic."}
    )


@dataclass
class ASEAgentConfig(AMPAgentConfig):
    """Main configuration class for ASE Agent."""

    _target_: str = "protomotions.agents.ase.agent.ASE"

    ase_parameters: ASEParametersConfig = field(
        default_factory=ASEParametersConfig,
        metadata={"help": "ASE-specific training parameters."}
    )
