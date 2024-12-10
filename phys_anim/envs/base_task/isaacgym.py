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

from isaacgym import gymtorch

from phys_anim.envs.base_task.common import BaseTask
from phys_anim.envs.humanoid.isaacgym import Humanoid


class TaskHumanoid(BaseTask, Humanoid):  # type: ignore[misc]
    ###############################################################
    # Set up IsaacGym environment
    ###############################################################
    def build_env(self, env_id, env_ptr, humanoid_asset):
        super().build_env(env_id, env_ptr, humanoid_asset)
        self.build_env_task(env_id, env_ptr, humanoid_asset)

    def build_env_task(self, env_id, env_ptr, humanoid_asset):
        pass

    ###############################################################
    # Handle reset
    ###############################################################
    def reset_env_tensors(self, env_ids, object_ids=None):
        super().reset_env_tensors(env_ids, object_ids)

        env_ids_int32 = self.get_task_actor_ids_for_reset(env_ids)
        if env_ids_int32 is not None:
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32),
            )

    ###############################################################
    # Getters
    ###############################################################
    def get_task_actor_ids_for_reset(self, env_ids):
        return None

    ###############################################################
    # Helpers
    ###############################################################
    def render(self):
        super().render()

        if not self.headless:
            self.draw_task()
