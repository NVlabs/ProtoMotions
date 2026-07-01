# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MimicADD observation additions."""

from types import SimpleNamespace

import torch

from protomotions.agents.mimic import agent_add
from protomotions.agents.mimic.agent_add import MimicADD


def _state(body_pos):
    return SimpleNamespace(
        rigid_body_pos=body_pos,
        rigid_body_rot=torch.zeros(body_pos.shape[0], body_pos.shape[1], 4),
        rigid_body_vel=torch.zeros_like(body_pos),
        rigid_body_ang_vel=torch.zeros_like(body_pos),
    )


def test_mimic_add_adds_tracking_difference_observation(monkeypatch):
    num_envs = 2
    ref_pos = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
        ]
    )
    cur_pos = ref_pos - 0.5
    env = SimpleNamespace(
        motion_manager=SimpleNamespace(
            motion_times=torch.tensor([0.1, 0.2]),
            motion_ids=torch.tensor([1, 2]),
        ),
        motion_lib=SimpleNamespace(
            get_motion_state=lambda motion_ids, motion_times: _state(ref_pos.clone())
        ),
        get_spawn_to_ref_pose_offset_with_terrain_height_correction=lambda pos: torch.zeros_like(
            pos
        ),
        terrain=SimpleNamespace(
            get_ground_heights=lambda roots: roots[:, 2] * 0.0,
        ),
        simulator=SimpleNamespace(get_bodies_state=lambda: _state(cur_pos.clone())),
    )
    agent = MimicADD.__new__(MimicADD)
    agent.env = env
    agent.num_envs = num_envs

    monkeypatch.setattr(
        agent_add.AMP,
        "add_agent_info_to_obs",
        lambda self, obs: {**obs, "base_added": torch.ones(num_envs, 1)},
    )

    def fake_max_coords(body_pos, ground_height, **kwargs):
        assert kwargs["local_obs"] is False
        assert kwargs["root_height_obs"] is True
        assert kwargs["observe_contacts"] is False
        assert kwargs["body_contacts"].shape == (num_envs, 0)
        return body_pos.reshape(num_envs, -1) + ground_height[:, None]

    monkeypatch.setattr(
        agent_add,
        "compute_humanoid_max_coords_observations",
        fake_max_coords,
    )

    obs = agent.add_agent_info_to_obs({"obs": torch.zeros(num_envs, 1)})

    assert torch.equal(obs["base_added"], torch.ones(num_envs, 1))
    assert torch.equal(
        obs["mimic_target_poses_diff"],
        (ref_pos - cur_pos).reshape(num_envs, -1),
    )


def test_mimic_add_expert_disc_obs_adds_zero_tracking_diff_from_history(monkeypatch):
    agent = MimicADD.__new__(MimicADD)
    agent.device = torch.device("cpu")
    agent.env = SimpleNamespace(
        observation_manager=SimpleNamespace(
            observation_history_buffers={
                "max_coords_obs": SimpleNamespace(data=torch.zeros(4, 5))
            }
        )
    )
    monkeypatch.setattr(
        agent_add.AMP,
        "get_expert_disc_obs",
        lambda self, num_samples: {"max_coords_obs": torch.ones(num_samples, 40)},
    )

    obs = agent.get_expert_disc_obs(3)

    assert torch.equal(obs["mimic_target_poses_diff"], torch.zeros(3, 5))


def test_mimic_add_expert_disc_obs_infers_dim_from_expert_obs(monkeypatch):
    agent = MimicADD.__new__(MimicADD)
    agent.device = torch.device("cpu")
    agent.env = SimpleNamespace(
        observation_manager=SimpleNamespace(observation_history_buffers={})
    )
    monkeypatch.setattr(
        agent_add.AMP,
        "get_expert_disc_obs",
        lambda self, num_samples: {
            "historical_max_coords_obs": torch.ones(num_samples, 24)
        },
    )

    obs = agent.get_expert_disc_obs(2)

    assert torch.equal(obs["mimic_target_poses_diff"], torch.zeros(2, 3))
