# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-fast motion-shard transitions performed after evaluation."""

from typing import TYPE_CHECKING, Tuple

import torch
import torch.distributed as dist

from protomotions.agents.utils.distributed import raise_if_any_rank_failed
from protomotions.components.motion_lib import MotionFileSwitchMode

if TYPE_CHECKING:
    from protomotions.agents.base_agent.agent import BaseAgent


def _build_motion_lib_from_config(*args, **kwargs):
    from protomotions.utils.component_builder import build_motion_lib_from_config

    return build_motion_lib_from_config(*args, **kwargs)


def _any_rank_changed(local_changed: bool, device: torch.device) -> bool:
    if not dist.is_initialized():
        return local_changed

    changed = torch.tensor(local_changed, device=device, dtype=torch.long)
    dist.all_reduce(changed, op=dist.ReduceOp.MAX)
    return bool(changed.item())


def advance_motion_shard_after_evaluation(
    agent: "BaseAgent", restart: bool = False
) -> Tuple[bool, bool]:
    """Advance to the next motion shard, replacing it live when configured."""
    mode = agent.motion_lib.config.motion_file_switch_mode
    if mode is MotionFileSwitchMode.FIXED:
        return False, False

    next_cycle = agent.motion_shard_cycle + 1
    selection = None
    selection_error = None
    try:
        selection = agent.motion_lib.selection_for_cycle(next_cycle)
    except Exception as error:
        selection_error = error
    raise_if_any_rank_failed(selection_error, "Motion shard selection", agent.device)

    selected_path, selected_index = selection
    local_changed = (
        selected_path != agent.motion_lib.motion_file
        or selected_index != agent.motion_lib.motion_file_shard_index
    )
    if not _any_rank_changed(local_changed, agent.device):
        return False, False

    agent.motion_shard_cycle = next_cycle
    agent.save(checkpoint_name="last.ckpt")
    if mode is MotionFileSwitchMode.RESTART or restart:
        return True, False

    transition_error = None
    try:
        if local_changed:
            motion_lib_config = agent.motion_lib.config
            agent.motion_lib = None
            agent.motion_manager = None
            agent.env.motion_lib = None
            agent.env.motion_manager = None

            replacement = _build_motion_lib_from_config(
                motion_lib_config,
                agent.device,
                shard_cycle=next_cycle,
            )
            agent.env.install_motion_lib(replacement)
            agent.motion_lib = agent.env.motion_lib
            agent.motion_manager = agent.env.motion_manager
            agent._load_environment_checkpoint()
            agent.model.reset_rollout_context(
                torch.arange(agent.num_envs, device=agent.device, dtype=torch.long)
            )
            agent.current_rewards.zero_()
            agent.current_lengths.zero_()
    except Exception as error:
        transition_error = error
    raise_if_any_rank_failed(transition_error, "Live motion shard switch", agent.device)
    return False, local_changed
