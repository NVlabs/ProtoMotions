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


class PathGenerator:
    def __init__(self, config, device, num_envs, episode_dur, height_conditioned):
        self.config = config
        self.device = device

        self.height_conditioned = height_conditioned

        self.dt = episode_dur / (self.config.num_verts - 1)

        self.verts_flat = torch.zeros(
            (num_envs * self.config.num_verts, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.verts = self.verts_flat.view((num_envs, self.config.num_verts, 3))

        self.head_max = self.config.head_height_max
        self.head_min = (
            self.config.head_height_min
        )  # + clip((speed - 1), max=0) * 0.4 --> speed = 1.5, height = 1

        self.use_naive_path_generator = (
            self.config.use_naive_path_generator or not self.height_conditioned
        )

    def reset(self, env_ids, init_pos):
        init_pos = init_pos.clone()
        init_pos[..., 2] = torch.clip(
            init_pos[..., 2], min=self.head_min, max=self.head_max
        )

        n = len(env_ids)
        if n > 0:
            num_verts = self.get_num_verts()
            dtheta = (
                2 * torch.rand([n, num_verts - 1], device=self.device) - 1.0
            )  # Sample the angles at each waypoint
            dtheta *= self.config.dtheta_max * self.dt

            dtheta_sharp = np.pi * (
                2 * torch.rand([n, num_verts - 1], device=self.device) - 1.0
            )  # Sharp Angles Angle
            sharp_probs = self.config.sharp_turn_prob * torch.ones_like(dtheta)
            sharp_mask = torch.bernoulli(sharp_probs) == 1.0
            dtheta[sharp_mask] = dtheta_sharp[sharp_mask]

            if self.config.get('use_forward_path_only', False):
                dtheta[:, 0] = np.pi * torch.ones(n, device=self.device)  # straight path
            else:
                dtheta[:, 0] = np.pi * (
                    2 * torch.rand([n], device=self.device) - 1.0
                )  # Heading

            dspeed = 2 * torch.rand([n, num_verts - 1], device=self.device) - 1.0
            dspeed *= self.config.accel_max * self.dt
            dspeed[:, 0] = (
                self.config.start_speed_max - self.config.speed_min
            ) * torch.rand([n], device=self.device) + self.config.speed_min  # Speed

            dspeed_z = (2 * torch.rand([n, num_verts - 1], device=self.device) - 1.0)
            dspeed_z *= self.config.accel_z_max * self.dt

            speed_z = torch.zeros_like(dspeed_z)
            head_height = torch.zeros((n, num_verts), device=self.device)
            head_height[:, 0] = init_pos[:, 2]

            speed = torch.zeros_like(dspeed)
            speed[:, 0] = dspeed[:, 0]
            for i in range(num_verts - 1):
                if i > 0:
                    speed_z[:, i] = speed_z[:, i - 1] + dspeed_z[:, i]
                else:
                    speed_z[:, i] = dspeed_z[:, i]  # Initial velocity

                speed_z[:, i] = torch.clip(speed_z[:, i],
                                         -self.config.speed_z_max,
                                         self.config.speed_z_max)

                head_height[:, i + 1] = head_height[:, i] + speed_z[:, i] * self.dt
                head_height[:, i + 1] = torch.clip(head_height[:, i + 1],
                                                   min=self.head_min,
                                                   max=self.head_max)
                if self.use_naive_path_generator:
                    max_speed = self.config.speed_max
                else:
                    # clip the speed based on the current height. at 0.4 the max speed is 1.5. At height 1.2 the max speed should be 5. Linearly interpolate between these two values.
                    max_speed = torch.clip(
                        1
                        + (self.config.speed_max - 1)
                        * (head_height[:, i] - self.head_min)
                        / (1.2 - self.head_min),
                        min=self.config.speed_min,
                        max=self.config.speed_max,
                    )
                speed[:, i] = torch.clip(
                    torch.clip(
                        speed[:, i - 1] + dspeed[:, i], min=self.config.speed_min
                    ),
                    max=max_speed,
                )

            ################################################
            if self.config.fixed_path:
                dtheta[:, :] = 0  # ZL: Hacking to make everything 0
                dtheta[0, 0] = 0  # ZL: Hacking to create collision
                if len(dtheta) > 1:
                    dtheta[1, 0] = -np.pi  # ZL: Hacking to create collision
                speed[:] = (self.config.speed_min + self.config.speed_max) / 2
            ################################################

            if self.config.slow:
                speed[:] = speed / 4

            dtheta = torch.cumsum(dtheta, dim=-1)

            seg_len = speed * self.dt

            dpos = torch.stack(
                [torch.cos(dtheta), -torch.sin(dtheta), torch.zeros_like(dtheta)],
                dim=-1,
            )
            dpos *= seg_len.unsqueeze(-1)
            dpos[..., 0, 0:2] += init_pos[..., 0:2]
            vert_pos = torch.cumsum(dpos, dim=-2)

            self.verts[env_ids, 0, :] = init_pos[..., 0:3]
            self.verts[env_ids, 1:] = vert_pos
            self.verts[env_ids, :, 2] = head_height

    def get_num_verts(self):
        return self.verts.shape[1]

    def get_num_segs(self):
        return self.get_num_verts() - 1

    def get_num_envs(self):
        return self.verts.shape[0]

    def get_traj_duration(self):
        num_verts = self.get_num_verts()
        dur = num_verts * self.dt
        return dur

    def get_traj_verts(self, traj_id):
        return self.verts[traj_id]

    def calc_pos(self, traj_ids, times):
        traj_dur = self.get_traj_duration()
        num_verts = self.get_num_verts()
        num_segs = self.get_num_segs()

        traj_phase = torch.clip(times / traj_dur, 0.0, 1.0)
        seg_idx = traj_phase * num_segs
        seg_id0 = torch.floor(seg_idx).long()
        seg_id1 = torch.ceil(seg_idx).long()
        lerp = seg_idx - seg_id0

        pos0 = self.verts_flat[traj_ids * num_verts + seg_id0]
        pos1 = self.verts_flat[traj_ids * num_verts + seg_id1]

        lerp = lerp.unsqueeze(-1)
        pos = (1.0 - lerp) * pos0 + lerp * pos1

        return pos
