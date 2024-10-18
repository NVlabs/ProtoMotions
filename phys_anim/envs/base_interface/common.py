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


class BaseInterface(object):
    def __init__(
        self,
        config,
        device: torch.device,
    ):
        self.config = config
        self.device = device
        self.headless = config.headless

        self.num_envs = config.num_envs

        self.control_freq_inv = config.simulator.sim.control_freq_inv

    def get_obs_size(self):
        raise NotImplementedError

    def on_environment_ready(self):
        pass

    def step(self, actions):
        raise NotImplementedError

    def pre_physics_step(self, actions):
        raise NotImplementedError

    def reset(self, env_ids=None):
        raise NotImplementedError

    def physics_step(self):
        if self.isaac_pd:
            self.apply_pd_control()
        for i in range(self.control_freq_inv):
            if not self.isaac_pd:
                self.apply_motor_forces()
            self.simulate()

    def simulate(self):
        raise NotImplementedError

    def post_physics_step(self):
        raise NotImplementedError

    def on_epoch_end(self, current_epoch: int):
        pass

    def close(self):
        raise NotImplementedError
