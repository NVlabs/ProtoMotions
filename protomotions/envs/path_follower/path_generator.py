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
import numpy as np
import torch

from protomotions.envs.obs.config import PathGeneratorConfig


class PathGenerator:
    def __init__(
        self,
        config: PathGeneratorConfig,
        device: torch.device,
        num_envs: int,
        episode_dur: float,
        height_conditioned: bool,
    ):
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

            if self.config.use_forward_path_only:
                dtheta[:, 0] = np.pi * torch.ones(
                    n, device=self.device
                )  # straight path
            else:
                dtheta[:, 0] = np.pi * (
                    2 * torch.rand([n], device=self.device) - 1.0
                )  # Heading

            dspeed = 2 * torch.rand([n, num_verts - 1], device=self.device) - 1.0
            dspeed *= self.config.accel_max * self.dt
            dspeed[:, 0] = (
                self.config.start_speed_max - self.config.speed_min
            ) * torch.rand([n], device=self.device) + self.config.speed_min  # Speed

            dspeed_z = 2 * torch.rand([n, num_verts - 1], device=self.device) - 1.0
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

                speed_z[:, i] = torch.clip(
                    speed_z[:, i], -self.config.speed_z_max, self.config.speed_z_max
                )

                head_height[:, i + 1] = head_height[:, i] + speed_z[:, i] * self.dt
                head_height[:, i + 1] = torch.clip(
                    head_height[:, i + 1], min=self.head_min, max=self.head_max
                )
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
