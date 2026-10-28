# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Feed a reference sequence through the real rolling history for AMP tests."""

import torch

from protomotions.envs.obs.state_history_buffer import HistorySample, StateHistoryBuffer
from protomotions.envs.state import BodyState


def roll_reference_history(state, num_envs, num_steps, contacts):
    """Return previous states after stepping oldest to newest, not reset seeding."""
    sequence = BodyState(
        rigid_body_pos=state.rigid_body_pos,
        rigid_body_rot=state.rigid_body_rot,
        rigid_body_vel=state.rigid_body_vel,
        rigid_body_ang_vel=state.rigid_body_ang_vel,
        dof_pos=state.dof_pos,
        dof_vel=state.dof_vel,
        ground_heights=torch.zeros(num_envs * (num_steps + 1)),
        rigid_body_contacts=contacts,
    ).map(lambda value: value.reshape(num_envs, num_steps + 1, *value.shape[1:]))
    actions = torch.zeros(num_envs, 2)

    def sample(index):
        return HistorySample(
            {"privileged": sequence.map(lambda value: value[:, index])},
            actions,
            actions,
        )

    history = StateHistoryBuffer(num_steps, sample(num_steps))
    for index in reversed(range(num_steps)):
        history.push(sample(index))
    return history.capture().states["privileged"].map(lambda value: value[:, 1:])
