# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint preparation and inspection snapshots for per-motion exploration."""

import torch


def prepare_checkpoint(checkpoint, motion_index, actor, load_training_state):
    """Validate row identity/bounds or expand shared std for a weights-only start.

    Neither the input checkpoint nor the actor is modified. Validation happens
    before the agent loads any weights or optimizer state.
    """
    if load_training_state or checkpoint["model"]["_actor.logstd"].ndim == 2:
        if checkpoint["motion_index"] != motion_index.state_dict():
            raise ValueError("Motion index differs from checkpoint; remap explicitly")
    else:
        # Weights-only warm start: every motion starts with the shared vector.
        # Optimizer/curriculum state starts fresh in the agent.
        checkpoint = dict(checkpoint, model=dict(checkpoint["model"]))
        source = checkpoint["model"]["_actor.logstd"]
        checkpoint["model"]["_actor.logstd"] = source.expand_as(actor.logstd).clone()

    logstd = checkpoint["model"]["_actor.logstd"]
    if not torch.all(
        (logstd >= actor.config.min_logstd) & (logstd <= actor.config.max_logstd)
    ):
        raise ValueError("Checkpoint logstd lies outside configured bounds")
    return checkpoint


def save_snapshot(path, motion_index, logstd, epoch, step_count):
    """Atomically replace the inspection sidecar; resume uses the full checkpoint.

    The agent owns rank-zero selection and distributed error propagation.
    """
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "motion_index": motion_index.state_dict(),
            "epoch": epoch,
            "step_count": step_count,
            "logstd": logstd.detach().cpu(),
        },
        temporary,
    )
    temporary.replace(path)
