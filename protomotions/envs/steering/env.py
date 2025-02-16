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

import numpy as np
import torch
from torch import Tensor
from isaac_utils import rotations, torch_utils
from protomotions.envs.base_env.env import BaseEnv
from protomotions.simulator.base_simulator.config import MarkerConfig, VisualizationMarker, MarkerState


class Steering(BaseEnv):
    def __init__(self, config, device: torch.device, *args, **kwargs):
        super().__init__(config=config, device=device, *args, **kwargs)

        self._tar_speed_min = self.config.steering_params.tar_speed_min
        self._tar_speed_max = self.config.steering_params.tar_speed_max

        self._heading_change_steps_min = (
            self.config.steering_params.heading_change_steps_min
        )
        self._heading_change_steps_max = (
            self.config.steering_params.heading_change_steps_max
        )
        self._random_heading_probability = (
            self.config.steering_params.random_heading_probability
        )
        self._standard_heading_change = (
            self.config.steering_params.standard_heading_change
        )
        self._standard_speed_change = self.config.steering_params.standard_speed_change
        self._stop_probability = self.config.steering_params.stop_probability

        self.steering_obs = torch.zeros(
            (self.config.num_envs, self.config.steering_params.obs_size),
            device=device,
            dtype=torch.float,
        )

        self._heading_change_steps = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.int64
        )
        self._prev_root_pos = torch.zeros(
            [self.num_envs, 3], device=self.device, dtype=torch.float
        )

        self._tar_dir_theta = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self._tar_dir = torch.zeros(
            [self.num_envs, 2], device=self.device, dtype=torch.float
        )
        self._tar_dir[..., 0] = 1.0

        self._tar_speed = torch.ones(
            [self.num_envs], device=self.device, dtype=torch.float
        )

    def create_visualization_markers(self):
        if self.config.headless:
            return {}

        visualization_markers = super().create_visualization_markers()
        
        steering_markers = []
        steering_markers.append(MarkerConfig(size="regular"))
        steering_markers_cfg = VisualizationMarker(
            type="arrow",
            color=(0.0, 1.0, 1.0),
            markers=steering_markers
        )
        visualization_markers["steering_markers"] = steering_markers_cfg

        return visualization_markers

    def get_markers_state(self):
        if self.config.headless:
            return {}

        markers_state = super().get_markers_state()

        marker_root_pos = self.simulator.get_root_state().root_pos
        marker_root_pos[..., 0:2] += self._tar_dir

        heading_axis = torch.zeros_like(marker_root_pos)
        heading_axis[..., -1] = 1.0
        marker_rot = rotations.quat_from_angle_axis(
            self._tar_dir_theta, heading_axis, True
        )
        markers_state["steering_markers"] = MarkerState(
            translation=marker_root_pos.view(self.num_envs, -1, 3),
            orientation=marker_rot.view(self.num_envs, -1, 4),
        )

        return markers_state

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if len(env_ids) > 0:
            self.reset_heading_task(env_ids)
        return super().reset(env_ids)

    def post_physics_step(self):
        super().post_physics_step()
        self.check_update_task()

    def check_update_task(self):
        reset_task_mask = self.progress_buf >= self._heading_change_steps
        rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        if len(rest_env_ids) > 0:
            self.reset_heading_task(rest_env_ids)

    def reset_heading_task(self, env_ids):
        n = len(env_ids)
        if np.random.binomial(1, self._random_heading_probability):
            dir_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
            tar_speed = (self._tar_speed_max - self._tar_speed_min) * torch.rand(
                n, device=self.device
            ) + self._tar_speed_min
        else:
            dir_delta_theta = (
                2 * self._standard_heading_change * torch.rand(n, device=self.device)
                - self._standard_heading_change
            )
            # map tar_dir_theta back to [0, 2pi], add delta, project back into [0, 2pi] and then shift.
            dir_theta = (dir_delta_theta + self._tar_dir_theta[env_ids] + np.pi) % (
                2 * np.pi
            ) - np.pi

            speed_delta = (
                2 * self._standard_speed_change * torch.rand(n, device=self.device)
                - self._standard_speed_change
            )
            tar_speed = torch.clamp(
                speed_delta + self._tar_speed[env_ids],
                min=self._tar_speed_min,
                max=self._tar_speed_max,
            )

        tar_dir = torch.stack([torch.cos(dir_theta), torch.sin(dir_theta)], dim=-1)

        change_steps = torch.randint(
            low=self._heading_change_steps_min,
            high=self._heading_change_steps_max,
            size=(n,),
            device=self.device,
            dtype=torch.int64,
        )

        stop_probs = torch.ones(n, device=self.device) * self._stop_probability
        should_stop = torch.bernoulli(stop_probs)

        self._tar_speed[env_ids] = tar_speed * (1.0 - should_stop)
        self._tar_dir_theta[env_ids] = dir_theta
        self._tar_dir[env_ids] = tar_dir
        self._heading_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps

    def compute_observations(self, env_ids=None):
        super().compute_observations(env_ids)

        if env_ids is None:
            root_states = self.simulator.get_root_state()
            tar_dir = self._tar_dir
            tar_speed = self._tar_speed
        else:
            root_states = self.simulator.get_root_state(env_ids)
            tar_dir = self._tar_dir[env_ids]
            tar_speed = self._tar_speed[env_ids]

        obs = compute_heading_observations(root_states.root_rot, tar_dir, tar_speed)
        self.steering_obs[env_ids] = obs

    def get_obs(self):
        obs = super().get_obs()
        obs.update({"steering": self.steering_obs})
        return obs

    def compute_reward(self):
        root_pos = self.simulator.get_root_state().root_pos
        self.rew_buf[:] = compute_heading_reward(
            root_pos, self._prev_root_pos, self._tar_dir, self._tar_speed, self.dt
        )
        self._prev_root_pos[:] = root_pos


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_heading_observations(
    root_rot: Tensor, tar_dir: Tensor, tar_speed: Tensor
) -> Tensor:
    tar_dir3d = torch.cat([tar_dir, torch.zeros_like(tar_dir[..., 0:1])], dim=-1)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot, True)

    local_tar_dir = rotations.quat_rotate(heading_rot, tar_dir3d, True)
    local_tar_dir = local_tar_dir[..., 0:2]

    tar_speed = tar_speed.unsqueeze(-1)

    obs = torch.cat([local_tar_dir, tar_speed], dim=-1)
    return obs


@torch.jit.script
def compute_heading_reward(
    root_pos: Tensor,
    prev_root_pos: Tensor,
    tar_dir: Tensor,
    tar_speed: Tensor,
    dt: float,
) -> Tensor:
    vel_err_scale = 0.25
    tangent_err_w = 0.1

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)

    tar_dir_vel = tar_dir_speed.unsqueeze(-1) * tar_dir
    tangent_vel = root_vel[..., :2] - tar_dir_vel

    tangent_speed = torch.sum(tangent_vel, dim=-1)

    tar_vel_err = tar_speed - tar_dir_speed
    tangent_vel_err = tangent_speed
    dir_reward = torch.exp(
        -vel_err_scale
        * (
            tar_vel_err * tar_vel_err
            + tangent_err_w * tangent_vel_err * tangent_vel_err
        )
    )

    speed_mask = tar_dir_speed < -0.5
    dir_reward[speed_mask] = 0

    return dir_reward
