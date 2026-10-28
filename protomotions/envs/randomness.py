# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The env's random streams, one per stage.

Every random draw on the env path (a control's command sampling, the reset
noise, the recovery swap, the motion sampling) asks the :class:`Randomness`
owner for its stage's generator and passes it as ``generator=``. Each stage
has its own generator, seeded from the initial seed and the stage name
(:func:`derive_seed`), so turning one stage's feature on or off leaves the
other stages' draws bit-identical, two runs at one seed draw the same
noise in every stage, and the generators save and restore with the env
snapshot, so an evaluation does not move training's streams. The entry
script folds the rank into the seed (``seed_everything(args.seed + rank)``),
so the owner is rank-agnostic. The simulator's, the sensor runtime's and the
terrain's draws are not this owner's.
"""

from __future__ import annotations

import hashlib
from typing import Dict

import torch
from torch import Tensor

STAGES = ("control", "reset_noise", "recovery", "motion_sampling")


def derive_seed(seed: int, stage: str) -> int:
    """The seed of ``stage``'s generator: the first eight bytes of
    ``sha256(f"{seed}:{stage}")``, little-endian. A pure function of the two,
    so the same seed and stage give the same stream on every machine, and
    two stages (or two ranks, whose seeds differ) never share one."""
    if stage not in STAGES:
        raise ValueError(f"unknown randomness stage {stage!r}; stages are {STAGES}")
    digest = hashlib.sha256(f"{seed}:{stage}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


class Randomness:
    """The per-stage generators, a snapshot owner (``env.randomness``)."""

    name = "randomness"

    def __init__(self, seed: int, device: torch.device):
        self.seed = seed
        self.device = device
        self._generators: Dict[str, torch.Generator] = {}

    def generator(self, stage: str) -> torch.Generator:
        """``stage``'s generator, created and seeded on first lookup."""
        if stage not in self._generators:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(derive_seed(self.seed, stage))
            self._generators[stage] = generator
        return self._generators[stage]

    def save_state(self) -> Dict[str, Tensor]:
        """The state of every stage's generator (created here if a stage has
        not drawn yet, so a snapshot always carries all four)."""
        return {stage: self.generator(stage).get_state() for stage in STAGES}

    def load_state(self, snapshot: Dict[str, Tensor]) -> None:
        for stage, state in snapshot.items():
            self.generator(stage).set_state(state)


__all__ = ["Randomness", "STAGES", "derive_seed"]
