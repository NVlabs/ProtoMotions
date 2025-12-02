# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import logging
import torch

from protomotions.agents.amp.agent import AMP
from protomotions.envs.mimic.env import Mimic
from protomotions.envs.utils.humanoid import compute_humanoid_max_coords_observations
from lightning.fabric import Fabric
from typing import Optional
from pathlib import Path

log = logging.getLogger(__name__)


class MimicADD(AMP):
    env: Mimic

    def __init__(
        self, fabric: Fabric, env: Mimic, config, root_dir: Optional[Path] = None
    ):
        super().__init__(fabric, env, config, root_dir)

    # -----------------------------
    # Environment Interaction and Data Updates
    # -----------------------------
    def add_agent_info_to_obs(self, obs):
        obs = super().add_agent_info_to_obs(obs)

        motion_times = self.env.motion_manager.motion_times
        motion_ids = self.env.motion_manager.motion_ids
        ref_state = self.env.motion_lib.get_motion_state(motion_ids, motion_times)

        ref_state_gt = ref_state.rigid_body_pos.reshape(self.num_envs, -1, 3)
        ref_state_gt += (
            self.env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
                ref_state_gt
            )
        )
        ref_ground_heights = self.env.terrain.get_ground_heights(
            ref_state_gt[:, 0]
        ).clone()

        current_state = self.env.simulator.get_bodies_state()
        ground_heights = self.env.terrain.get_ground_heights(
            current_state.rigid_body_pos[:, 0]
        ).clone()

        local_obs = False
        root_height_obs = self.env.self_obs_cb.config.max_coords_obs.root_height_obs
        observe_contacts = self.env.self_obs_cb.config.max_coords_obs.observe_contacts
        assert (
            observe_contacts is False
        ), "ADD does not yet support contact based conditioning"

        # Empty contact flags since observe_contacts is False
        empty_contacts = torch.zeros(
            self.num_envs, 0, dtype=torch.bool, device=ref_state_gt.device
        )

        ref_pose = compute_humanoid_max_coords_observations(
            body_pos=ref_state_gt,
            body_rot=ref_state.rigid_body_rot,
            body_vel=ref_state.rigid_body_vel,
            body_ang_vel=ref_state.rigid_body_ang_vel,
            ground_height=ref_ground_heights,
            body_contacts=empty_contacts,
            local_obs=local_obs,
            root_height_obs=root_height_obs,
            observe_contacts=observe_contacts,
            w_last=True,
        )

        current_pose = compute_humanoid_max_coords_observations(
            body_pos=current_state.rigid_body_pos,
            body_rot=current_state.rigid_body_rot,
            body_vel=current_state.rigid_body_vel,
            body_ang_vel=current_state.rigid_body_ang_vel,
            ground_height=ground_heights,
            body_contacts=empty_contacts,
            local_obs=local_obs,
            root_height_obs=root_height_obs,
            observe_contacts=observe_contacts,
            w_last=True,
        )

        tracking_diff_obs = ref_pose - current_pose
        obs["mimic_target_poses_diff"] = tracking_diff_obs.view(self.num_envs, -1)
        return obs

    def get_expert_disc_obs(self, num_samples: int):
        expert_disc_obs = super().get_expert_disc_obs(num_samples)
        tracking_diff_obs = torch.zeros(
            [num_samples, self.env.self_obs_cb.humanoid_max_coords_obs.shape[-1]],
            device=self.device,
        )
        expert_disc_obs["mimic_target_poses_diff"] = tracking_diff_obs

        return expert_disc_obs
