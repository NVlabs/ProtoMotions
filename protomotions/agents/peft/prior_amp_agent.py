# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discrete-prior PEFT RLFT agent with AMP discriminator rewards."""

from pathlib import Path
from typing import Optional

from lightning.fabric import Fabric

from protomotions.agents.amp.component import AMPAgentMixin
from protomotions.agents.peft.prior_agent import DiscretePriorPEFTRLFTAgent


class DiscretePriorPEFTRLFTAMPAgent(AMPAgentMixin, DiscretePriorPEFTRLFTAgent):
    """Discrete-prior PEFT RLFT with AMP-style reward shaping."""

    def __init__(self, fabric: Fabric, env, config, root_dir: Optional[Path] = None):
        if config.model.critic is None:
            raise ValueError(
                "DiscretePriorPEFTRLFTAMPAgent requires config.model.critic."
            )
        super().__init__(fabric, env, config, root_dir=root_dir)

    def _load_parameters_without_amp_component(
        self,
        state_dict,
        load_training_state: bool = True,
    ):
        DiscretePriorPEFTRLFTAgent.load_parameters(
            self,
            state_dict,
            load_training_state=load_training_state,
        )
