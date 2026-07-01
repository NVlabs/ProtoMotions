# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AMP discriminator reference-observation parity with simulator reset history."""

from types import SimpleNamespace

import torch

from protomotions.agents.amp.component import AMPTrainingComponent
from protomotions.agents.amp.agent import AMP
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.obs.humanoid_historical import (
    compute_historical_max_coords_from_motion_lib,
    compute_historical_max_coords_from_state,
)
from protomotions.envs.obs.state_history_buffer import StateHistoryBuffer


def _new_amp_agent():
    agent = object.__new__(AMP)
    component = object.__new__(AMPTrainingComponent)
    component.agent = agent
    agent.amp_component = component
    return agent


class _FixedMotionManager:
    def __init__(self, motion_ids, motion_times):
        self.motion_ids = motion_ids
        self.motion_times = motion_times

    def sample_n_motion_ids(self, num_samples):
        assert num_samples == self.motion_ids.shape[0]
        return self.motion_ids

    def sample_time(self, motion_ids):
        assert torch.equal(motion_ids, self.motion_ids)
        return self.motion_times


class _ContactMotionLib:
    num_bodies = 4
    num_dofs = 3

    def __init__(self):
        self.motion_lengths = torch.full((8,), 10.0)

    def get_motion_state(self, motion_ids, motion_times):
        batch = motion_ids.shape[0]
        ids = motion_ids.float().view(batch, 1, 1)
        times = motion_times.view(batch, 1, 1)
        body_ids = torch.arange(self.num_bodies, dtype=torch.float).view(
            1, self.num_bodies, 1
        )

        rigid_body_pos = torch.cat(
            [
                ids * 0.3 + body_ids * 0.2 + times,
                ids * -0.2 + body_ids * 0.4 + times * 0.1,
                0.5 + body_ids * 0.15 + times * 0.25,
            ],
            dim=-1,
        )
        rigid_body_rot = torch.zeros(batch, self.num_bodies, 4)
        rigid_body_rot[..., 3] = 1.0
        rigid_body_vel = torch.cat(
            [
                times.expand(batch, self.num_bodies, 1),
                (ids + 0.25).expand(batch, self.num_bodies, 1),
                (body_ids + 0.5).expand(batch, self.num_bodies, 1),
            ],
            dim=-1,
        )
        rigid_body_ang_vel = torch.cat(
            [
                (body_ids + 0.75).expand(batch, self.num_bodies, 1),
                (times + 1.0).expand(batch, self.num_bodies, 1),
                (ids + 1.25).expand(batch, self.num_bodies, 1),
            ],
            dim=-1,
        )
        dof_base = torch.arange(self.num_dofs, dtype=torch.float).view(1, self.num_dofs)
        dof_pos = motion_ids.float().unsqueeze(-1) + motion_times.unsqueeze(-1) + dof_base
        dof_vel = motion_times.unsqueeze(-1) * 0.5 + dof_base

        body_indices = torch.arange(self.num_bodies).view(1, self.num_bodies)
        rigid_body_contacts = (
            (
                motion_ids.view(batch, 1)
                + body_indices
                + (motion_times * 10).long().view(batch, 1)
            )
            % 2
            == 0
        )

        return SimpleNamespace(
            rigid_body_pos=rigid_body_pos,
            rigid_body_rot=rigid_body_rot,
            rigid_body_vel=rigid_body_vel,
            rigid_body_ang_vel=rigid_body_ang_vel,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            rigid_body_contacts=rigid_body_contacts,
        )


class _NoLengthsNoContactsMotionLib:
    num_bodies = 3

    def get_motion_state(self, motion_ids, motion_times):
        batch = motion_ids.shape[0]
        ids = motion_ids.float().view(batch, 1, 1)
        times = motion_times.view(batch, 1, 1)
        body_ids = torch.arange(self.num_bodies, dtype=torch.float).view(
            1, self.num_bodies, 1
        )

        rigid_body_pos = torch.cat(
            [
                ids + body_ids * 0.2 + times,
                ids * -0.3 + body_ids * 0.1 + times * 0.5,
                0.4 + body_ids * 0.25 + times * 0.2,
            ],
            dim=-1,
        )
        rigid_body_rot = torch.zeros(batch, self.num_bodies, 4)
        rigid_body_rot[..., 3] = 1.0
        rigid_body_vel = torch.cat(
            [
                times.expand(batch, self.num_bodies, 1),
                (ids + 1.5).expand(batch, self.num_bodies, 1),
                (body_ids + 2.5).expand(batch, self.num_bodies, 1),
            ],
            dim=-1,
        )
        rigid_body_ang_vel = torch.cat(
            [
                (body_ids + 3.5).expand(batch, self.num_bodies, 1),
                (times + 4.5).expand(batch, self.num_bodies, 1),
                (ids + 5.5).expand(batch, self.num_bodies, 1),
            ],
            dim=-1,
        )

        return SimpleNamespace(
            rigid_body_pos=rigid_body_pos,
            rigid_body_rot=rigid_body_rot,
            rigid_body_vel=rigid_body_vel,
            rigid_body_ang_vel=rigid_body_ang_vel,
        )


def _simulator_history_obs_from_reference_reset(
    motion_lib,
    motion_ids,
    motion_times,
    *,
    dt,
    num_state_history_steps,
    contact_body_ids,
    history_steps,
):
    num_envs = motion_ids.shape[0]
    buffer_size = num_state_history_steps + 1
    time_offsets = -dt * torch.arange(buffer_size)
    expanded_motion_ids = motion_ids.unsqueeze(1).expand(-1, buffer_size)
    expanded_motion_times = motion_times.unsqueeze(1) + time_offsets.unsqueeze(0)
    expanded_motion_times = expanded_motion_times.clamp(min=0.0)
    expanded_motion_times = torch.min(
        expanded_motion_times,
        motion_lib.motion_lengths[motion_ids].unsqueeze(1).expand(-1, buffer_size),
    )

    historical_state = motion_lib.get_motion_state(
        expanded_motion_ids.reshape(-1),
        expanded_motion_times.reshape(-1),
    )

    history = StateHistoryBuffer(
        num_envs=num_envs,
        num_history_steps=num_state_history_steps,
        num_bodies=motion_lib.num_bodies,
        num_dofs=motion_lib.num_dofs,
        action_dim=2,
        num_contact_bodies=len(contact_body_ids),
        anchor_body_index=0,
        device=torch.device("cpu"),
    )
    history.reset_from_states(
        env_ids=torch.arange(num_envs),
        rigid_body_pos=historical_state.rigid_body_pos.view(
            num_envs, buffer_size, motion_lib.num_bodies, 3
        ),
        rigid_body_rot=historical_state.rigid_body_rot.view(
            num_envs, buffer_size, motion_lib.num_bodies, 4
        ),
        rigid_body_vel=historical_state.rigid_body_vel.view(
            num_envs, buffer_size, motion_lib.num_bodies, 3
        ),
        rigid_body_ang_vel=historical_state.rigid_body_ang_vel.view(
            num_envs, buffer_size, motion_lib.num_bodies, 3
        ),
        dof_pos=historical_state.dof_pos.view(
            num_envs, buffer_size, motion_lib.num_dofs
        ),
        dof_vel=historical_state.dof_vel.view(
            num_envs, buffer_size, motion_lib.num_dofs
        ),
        ground_heights=torch.zeros(num_envs, buffer_size),
        body_contacts=historical_state.rigid_body_contacts[:, contact_body_ids]
        .view(num_envs, buffer_size, -1)
        .bool(),
    )

    return compute_historical_max_coords_from_state(
        historical_rigid_body_pos=history.historical_rigid_body_pos,
        historical_rigid_body_rot=history.historical_rigid_body_rot,
        historical_rigid_body_vel=history.historical_rigid_body_vel,
        historical_rigid_body_ang_vel=history.historical_rigid_body_ang_vel,
        historical_ground_heights=history.historical_ground_heights,
        historical_body_contacts=history.historical_body_contacts,
        local_obs=False,
        root_height_obs=True,
        observe_contacts=True,
        w_last=True,
        history_steps=history_steps,
    )


def test_amp_contact_reference_obs_matches_simulator_reference_reset_history():
    motion_ids = torch.tensor([0, 3])
    motion_times = torch.tensor([1.1, 1.7])
    dt = 0.1
    num_state_history_steps = 4
    history_steps = [1, 2, 4]
    contact_body_ids = torch.tensor([1, 3])
    motion_lib = _ContactMotionLib()

    agent = _new_amp_agent()
    agent.motion_manager = _FixedMotionManager(motion_ids, motion_times)
    agent.motion_lib = motion_lib
    agent.num_envs = 1
    agent.env = SimpleNamespace(
        simulator=SimpleNamespace(dt=dt),
        config=SimpleNamespace(num_state_history_steps=num_state_history_steps),
        contact_body_ids=contact_body_ids,
    )
    agent.config = SimpleNamespace(
        reference_obs_components={
            "historical_max_coords_obs": MdpComponent(
                compute_func=compute_historical_max_coords_from_motion_lib,
                dynamic_vars={},
                static_params={
                    "history_steps": history_steps,
                    "local_obs": False,
                    "root_height_obs": True,
                    "observe_contacts": True,
                },
            )
        }
    )
    agent.amp_component.discriminator = SimpleNamespace(
        module=SimpleNamespace(in_keys=["historical_max_coords_obs"])
    )

    expert_obs = AMP.get_expert_disc_obs(agent, num_samples=motion_ids.shape[0])
    simulator_obs = _simulator_history_obs_from_reference_reset(
        motion_lib,
        motion_ids,
        motion_times,
        dt=dt,
        num_state_history_steps=num_state_history_steps,
        contact_body_ids=contact_body_ids,
        history_steps=history_steps,
    )

    assert torch.allclose(
        expert_obs["historical_max_coords_obs"],
        simulator_obs,
        atol=1e-6,
    )


def test_amp_reference_obs_chunks_missing_lengths_and_contact_data():
    motion_ids = torch.tensor([0, 1, 2, 3, 4])
    motion_times = torch.tensor([0.15, 0.35, 0.55, 0.75, 0.95])
    dt = 0.1
    num_state_history_steps = 3
    history_steps = [1, 3]
    contact_body_ids = torch.tensor([0, 2])
    motion_lib = _NoLengthsNoContactsMotionLib()

    agent = _new_amp_agent()
    agent.motion_manager = _FixedMotionManager(motion_ids, motion_times)
    agent.motion_lib = motion_lib
    agent.num_envs = 2
    agent.env = SimpleNamespace(
        simulator=SimpleNamespace(dt=dt),
        config=SimpleNamespace(num_state_history_steps=num_state_history_steps),
        contact_body_ids=contact_body_ids,
    )
    agent.config = SimpleNamespace(
        reference_obs_components={
            "historical_max_coords_obs": MdpComponent(
                compute_func=compute_historical_max_coords_from_motion_lib,
                dynamic_vars={},
                static_params={
                    "history_steps": history_steps,
                    "local_obs": False,
                    "root_height_obs": True,
                    "observe_contacts": True,
                },
            )
        }
    )
    agent.amp_component.discriminator = SimpleNamespace(
        module=SimpleNamespace(in_keys=["historical_max_coords_obs"])
    )

    expert_obs = AMP.get_expert_disc_obs(agent, num_samples=motion_ids.shape[0])
    expected = compute_historical_max_coords_from_motion_lib(
        motion_lib,
        motion_ids=motion_ids,
        motion_times=motion_times,
        num_state_history_steps=num_state_history_steps,
        dt=dt,
        local_obs=False,
        root_height_obs=True,
        observe_contacts=True,
        contact_body_ids=contact_body_ids,
        history_steps=history_steps,
    )

    assert torch.allclose(
        expert_obs["historical_max_coords_obs"],
        expected,
        atol=1e-6,
    )
    assert torch.equal(
        expert_obs["historical_max_coords_obs"][:, -2:],
        torch.zeros(motion_ids.shape[0], contact_body_ids.numel()),
    )
