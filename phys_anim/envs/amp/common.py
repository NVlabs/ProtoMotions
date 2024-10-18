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

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from phys_anim.envs.humanoid.humanoid_utils import build_disc_observations
from phys_anim.envs.env_utils.general import HistoryBuffer

if TYPE_CHECKING:
    from phys_anim.envs.amp.isaacgym import DiscHumanoid
else:
    DiscHumanoid = object


class BaseDisc(DiscHumanoid):  # type: ignore[misc]
    def __init__(self, config, device: torch.device):
        super().__init__(config, device)

        if not self.config.disable_discriminator:
            self.discriminator_obs_historical_steps = (
                self.config.discriminator_obs_historical_steps
            )
            assert self.discriminator_obs_historical_steps >= 2

            self.disc_hist_buf = HistoryBuffer(
                self.discriminator_obs_historical_steps,
                self.num_envs,
                shape=(self.discriminator_obs_size_per_step,),
                device=self.device,
            )

    ###############################################################
    # Set up environment
    ###############################################################
    def setup_character_props(self):
        super().setup_character_props()
        if not self.config.disable_discriminator:
            self.discriminator_obs_size_per_step = (
                self.config.discriminator_obs_size_per_step
            )

    ###############################################################
    # Environment step logic
    ###############################################################
    def post_physics_step(self):
        super().post_physics_step()
        if not self.config.disable_discriminator:
            self.disc_hist_buf.rotate()
            self.compute_disc_observations()

            self.extras["disc_obs"] = self.make_disc_obs()

    ###############################################################
    # Getters
    ###############################################################
    def get_num_disc_obs(self):
        return (
            self.discriminator_obs_historical_steps
            * self.discriminator_obs_size_per_step
        )

    def get_required_history_length(self):
        if self.config.disable_discriminator:
            return super().get_required_history_length()
        else:
            return self.discriminator_obs_historical_steps

    ###############################################################
    # Handle Resets
    ###############################################################
    def reset_envs(self, env_ids):
        if len(env_ids) == 0:
            return

        super().reset_envs(env_ids)

        if not self.config.disable_discriminator:
            self.reset_disc_hist_buf(env_ids)

    def reset_disc_hist_buf(self, env_ids):
        self.compute_disc_observations(env_ids)

        if len(self.reset_default_env_ids) > 0:
            self.reset_disc_hist_default(self.reset_default_env_ids)

        if len(self.reset_ref_env_ids) > 0:
            self.reset_disc_hist_ref(
                self.reset_ref_env_ids,
                self.reset_ref_motion_ids,
                self.reset_ref_motion_times,
            )

    def reset_disc_hist_default(self, env_ids):
        self.disc_hist_buf.set_hist(
            self.disc_hist_buf.get_current(env_ids), env_ids=env_ids
        )

    def reset_disc_hist_ref(self, env_ids, motion_ids, motion_times):
        dt = self.dt
        motion_ids = torch.tile(
            motion_ids.unsqueeze(-1), [1, self.discriminator_obs_historical_steps - 1]
        )
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (
            torch.arange(
                0, self.discriminator_obs_historical_steps - 1, device=self.device
            )
            + 1
        )
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)

        ref_state = self.motion_lib.get_motion_state(motion_ids, motion_times)

        disc_obs_demo = build_disc_observations(
            ref_state.root_pos,
            ref_state.root_rot,
            ref_state.root_vel,
            ref_state.root_ang_vel,
            ref_state.dof_pos,
            ref_state.dof_vel,
            ref_state.key_body_pos,
            torch.zeros(len(motion_ids), 1, device=self.device),
            self.config.humanoid_obs.local_root_obs,
            self.config.humanoid_obs.root_height_obs,
            self.dof_obs_size,
            self.get_dof_offsets(),
            False,
            self.w_last,
        )
        self.disc_hist_buf.set_hist(
            disc_obs_demo.view(
                len(env_ids), self.discriminator_obs_historical_steps - 1, -1
            ).permute(1, 0, 2),
            env_ids,
        )

    ###############################################################
    # Helpers
    ###############################################################
    def make_disc_obs(self):
        return self.disc_hist_buf.get_all_flattened()

    def build_disc_obs_demo(self, motion_ids: Tensor, motion_times0: Tensor):
        dt = self.dt

        motion_ids = torch.tile(
            motion_ids.unsqueeze(-1), [1, self.discriminator_obs_historical_steps]
        )
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(
            0, self.discriminator_obs_historical_steps, device=self.device
        )
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)

        # motion ids above are "sub_motions" so first we map to motion file itself and then extract the length.
        lengths = self.motion_lib.state.motion_lengths[
            self.motion_lib.state.sub_motion_to_motion[motion_ids]
        ]

        assert torch.all(motion_times >= 0)
        assert torch.all(motion_times <= lengths)

        ref_state = self.motion_lib.get_motion_state(motion_ids, motion_times)

        disc_obs_demo = build_disc_observations(
            ref_state.root_pos,
            ref_state.root_rot,
            ref_state.root_vel,
            ref_state.root_ang_vel,
            ref_state.dof_pos,
            ref_state.dof_vel,
            ref_state.key_body_pos,
            torch.zeros(len(motion_ids), 1, device=self.device),
            self.config.humanoid_obs.local_root_obs,
            self.config.humanoid_obs.root_height_obs,
            self.dof_obs_size,
            self.get_dof_offsets(),
            False,
            self.w_last,
        )
        return disc_obs_demo

    def compute_disc_observations(self, env_ids=None):
        current_state = self.get_bodies_state()

        dof_pos, dof_vel = self.get_dof_state()
        key_body_pos = current_state.body_pos[:, self.key_body_ids, :]

        if env_ids is None:
            disc_obs = build_disc_observations(
                current_state.body_pos[:, 0, :],
                current_state.body_rot[:, 0, :],
                current_state.body_vel[:, 0, :],
                current_state.body_ang_vel[:, 0, :],
                dof_pos,
                dof_vel,
                key_body_pos,
                self.get_ground_heights(current_state.body_pos[:, 0, :2]),
                self.config.humanoid_obs.local_root_obs,
                self.config.humanoid_obs.root_height_obs,
                self.dof_obs_size,
                self.get_dof_offsets(),
                False,
                self.w_last,
            )
            self.disc_hist_buf.set_curr(disc_obs)
        else:
            disc_obs = build_disc_observations(
                current_state.body_pos[env_ids, 0, :],
                current_state.body_rot[env_ids, 0, :],
                current_state.body_vel[env_ids, 0, :],
                current_state.body_ang_vel[env_ids, 0, :],
                dof_pos[env_ids],
                dof_vel[env_ids],
                key_body_pos[env_ids],
                self.get_ground_heights(current_state.body_pos[:, 0, :2])[env_ids],
                self.config.humanoid_obs.local_root_obs,
                self.config.humanoid_obs.root_height_obs,
                self.dof_obs_size,
                self.get_dof_offsets(),
                False,
                self.w_last,
            )
            self.disc_hist_buf.set_curr(disc_obs, env_ids)
