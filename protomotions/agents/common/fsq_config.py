# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for finite-scalar-quantized autoencoders."""

from dataclasses import dataclass, field

from protomotions.agents.common.autoencoder.config import AutoEncoderConfig


@dataclass
class FSQAutoEncoderConfig(AutoEncoderConfig):
    """Finite scalar quantization autoencoder configuration."""

    _target_: str = "protomotions.agents.common.fsq.FSQAutoEncoder"
    num_fsq_levels: int = field(
        default=7, metadata={"help": "Number of quantization levels. Must be odd."}
    )
    num_fsq_scalars: int = field(
        default=24, metadata={"help": "Number of FSQ scalar code dimensions."}
    )
