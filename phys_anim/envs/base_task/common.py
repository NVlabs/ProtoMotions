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

if TYPE_CHECKING:
    from phys_anim.envs.base_task.isaacgym import TaskHumanoid
else:
    TaskHumanoid = object


class BaseTask(TaskHumanoid):  # type: ignore[misc]
    def __init__(self, config, device):
        super().__init__(config, device)
        self.setup_task()

    ###############################################################
    # Set up environment
    ###############################################################
    def setup_task(self):
        pass

    ###############################################################
    # Handle reset
    ###############################################################
    def reset_envs(self, env_ids):
        super().reset_envs(env_ids)
        self.reset_task(env_ids)

    def reset_task(self, env_ids):
        pass

    ###############################################################
    # Environment step logic
    ###############################################################
    def compute_observations(self, env_ids=None):
        super().compute_observations(env_ids)

        # After the humanoid obs is called, we compute the task obs.
        # A task obs will have its own unique tensor.
        # We do not append the task obs to the humanoid obs, rather we allow the user to define the network structure
        # and how the task obs is used in the network.
        self.compute_task_obs()

    def compute_task_obs(self, env_ids=None):
        pass

    def compute_reward(self, actions):
        return 0

    def pre_physics_step(self, actions):
        self.update_task(actions)
        super().pre_physics_step(actions)

    def update_task(self, actions):
        pass

    ###############################################################
    # Helpers
    ###############################################################
    def draw_task(self):
        return
