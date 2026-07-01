# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parameter-efficient fine-tuning configuration."""

from dataclasses import dataclass, field


@dataclass
class TransformerPEFTConfig:
    """Parameter-efficient adapter config for transformer layers."""

    peft_type: str = field(default="dora", metadata={"help": "'lora' or 'dora'."})
    rank: int = field(default=4, metadata={"help": "Adapter rank."})
    alpha: float = field(default=0.5, metadata={"help": "Adapter scaling factor."})
