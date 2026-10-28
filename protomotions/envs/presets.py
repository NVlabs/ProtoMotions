# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named environment presets a recipe's inference hook applies.

A preset is the one spelling of a change several recipes make the same way, so
an inference hook reads as its intent (``tracking_playback(configs.env)``)
rather than the lines that implement it. Presets mutate the passed config in
place and run on the inference view, never the training bundle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from protomotions.envs.base_env.config import EnvConfig


def tracking_playback(env: "EnvConfig") -> None:
    """Play a tracked clip through to its end from its first frame.

    Clears the terminations (an early fall must not cut the clip short), and
    makes every reset resample a clip and start it at the first frame. This is
    the shared body of the tracker and GPC recipes' inference hooks.
    """
    env.termination_components = {}
    env.motion_manager.resample_on_reset = True
    env.motion_manager.init_start_prob = 1.0
