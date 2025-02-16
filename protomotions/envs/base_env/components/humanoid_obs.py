# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
from torch import Tensor

from protomotions.envs.base_env.env_utils.humanoid_utils import (
    compute_humanoid_observations,
    compute_humanoid_observations_max,
)
from protomotions.envs.base_env.components.base_component import BaseComponent
from protomotions.envs.base_env.env_utils.general import HistoryBuffer


class HumanoidObs(BaseComponent):
    def __init__(self, config, env):
        super().__init__(config, env)
        self.humanoid_obs = torch.zeros(
            self.env.num_envs,
            self.config.obs_size,
            dtype=torch.float,
            device=self.env.device,
        )
        self.humanoid_obs_hist_buf = HistoryBuffer(
            self.config.num_historical_steps,
            self.env.num_envs,
            shape=(self.config.obs_size,),
            device=self.env.device,
        )
        body_names = self.env.config.robot.body_names
        num_bodies = len(body_names)
        self.body_contacts = torch.zeros(
            self.env.num_envs,
            num_bodies,
            3,
            dtype=torch.bool,
            device=self.env.device,
        )

    def post_physics_step(self):
        self.humanoid_obs_hist_buf.rotate()

    def reset_envs(self, env_ids, reset_default_env_ids, reset_ref_env_ids, reset_ref_motion_ids, reset_ref_motion_times):
        if self.config.num_historical_steps > 1:
            self.reset_hist_buf(env_ids, reset_default_env_ids, reset_ref_env_ids, reset_ref_motion_ids, reset_ref_motion_times)

    def reset_hist_buf(self, env_ids, reset_default_env_ids, reset_ref_env_ids, reset_ref_motion_ids, reset_ref_motion_times):
        if len(reset_default_env_ids) > 0:
            self.reset_hist_default(reset_default_env_ids)

        if len(reset_ref_env_ids) > 0:
            self.reset_hist_ref(
                reset_ref_env_ids,
                reset_ref_motion_ids,
                reset_ref_motion_times,
            )

    def reset_hist_default(self, env_ids):
        self.humanoid_obs_hist_buf.set_hist(
            self.humanoid_obs_hist_buf.get_current(env_ids), env_ids=env_ids
        )

    def reset_hist_ref(self, env_ids, motion_ids, motion_times):
        dt = self.env.dt
        motion_ids = torch.tile(
            motion_ids.unsqueeze(-1), [1, self.config.num_historical_steps - 1]
        )
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (
            torch.arange(
                0, self.config.num_historical_steps - 1, device=self.env.device
            )
            + 1
        )
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1).clamp(min=0)

        ref_state = self.env.motion_lib.get_motion_state(motion_ids, motion_times)

        obs_ref = compute_humanoid_observations_max(
            ref_state.rigid_body_pos,
            ref_state.rigid_body_rot,
            ref_state.rigid_body_vel,
            ref_state.rigid_body_ang_vel,
            torch.zeros(len(motion_ids), 1, device=self.env.device),
            self.config.local_root_obs,
            self.config.root_height_obs,
            True,
        )
        self.humanoid_obs_hist_buf.set_hist(
            obs_ref.view(
                len(env_ids), self.config.num_historical_steps - 1, -1
            ).permute(1, 0, 2),
            env_ids,
        )

    def compute_observations(self, env_ids):
        current_state = self.env.simulator.get_bodies_state(env_ids)
        body_contacts = self.env.simulator.get_bodies_contact_buf(env_ids)

        ground_heights = self.env.terrain.get_ground_heights(current_state.rigid_body_pos[:, 0]).clone()

        if self.config.use_max_coords_obs:
            obs = compute_humanoid_observations_max(
                current_state.rigid_body_pos,
                current_state.rigid_body_rot,
                current_state.rigid_body_vel,
                current_state.rigid_body_ang_vel,
                ground_heights,
                self.config.local_root_obs,
                self.config.root_height_obs,
                True,
            )

        else:
            dof_state = self.env.simulator.get_dof_state(env_ids)
            dof_pos = dof_state.dof_pos
            dof_vel = dof_state.dof_vel

            root_pos = current_state.rigid_body_pos[:, 0, :]
            root_rot = current_state.rigid_body_rot[:, 0, :]
            root_vel = current_state.rigid_body_vel[:, 0, :]
            root_ang_vel = current_state.rigid_body_ang_vel[:, 0, :]
            key_body_pos = current_state.rigid_body_pos[:, self.env.simulator.key_body_ids, :]

            obs = compute_humanoid_observations(
                root_pos,
                root_rot,
                root_vel,
                root_ang_vel,
                dof_pos,
                dof_vel,
                key_body_pos,
                ground_heights,
                self.config.local_root_obs,
                self.env.simulator.dof_obs_size,
                self.env.simulator.get_dof_offsets(),
                True,
            )
        self.body_contacts[:] = body_contacts
        self.humanoid_obs[env_ids] = obs
        self.humanoid_obs_hist_buf.set_curr(obs, env_ids)

    def build_self_obs_demo(
        self, motion_ids: Tensor, motion_times0: Tensor, num_steps: int
    ):
        dt = self.env.dt

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, num_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, num_steps, device=self.env.device)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)

        lengths = self.env.motion_lib.state.motion_lengths[motion_ids]

        motion_times = motion_times.view(-1).clamp(max=lengths).clamp(min=0)

        ref_state = self.env.motion_lib.get_motion_state(motion_ids, motion_times)

        obs_demo = compute_humanoid_observations_max(
            ref_state.rigid_body_pos,
            ref_state.rigid_body_rot,
            ref_state.rigid_body_vel,
            ref_state.rigid_body_ang_vel,
            torch.zeros(len(motion_ids), 1, device=self.env.device),
            self.config.local_root_obs,
            self.config.root_height_obs,
            True,
        )
        return obs_demo

    def get_obs(self):
        return {
            "self_obs": self.humanoid_obs.clone(),
            "historical_self_obs": self.humanoid_obs_hist_buf.get_all_flattened().clone(),
        }
