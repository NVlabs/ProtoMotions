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

import os

import carb
import torch
from omni.isaac.kit import SimulationApp

from phys_anim.envs.base_interface.common import BaseInterface


class SimBaseInterface(BaseInterface):
    def __init__(self, config, device: torch.device):
        """
        This class provides a unified interface with IsaacGym environments.
        The actual environments build on-top of the 'BaseTask' class.
        """
        experience = ""
        if config.headless:
            experience = (
                f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
            )

        device_str = str(device)
        rank = 0
        if "cuda" in device_str:
            rank = int(device_str.split(":")[-1])
        self._simulation_app = SimulationApp(
            {"headless": config.headless, "physics_gpu": rank}, experience=experience
        )
        carb.settings.get_settings().set(
            "/persistent/omnihydra/useSceneGraphInstancing", True
        )
        self._render = not config.headless

        super().__init__(config, device)

    def __getattr__(self, name):
        return self._task.__getattribute__(name)

    def set_task(self, task, sim_config) -> None:
        """Creates a World object and adds Task to World.
            Initializes and registers task to the environment interface.
            Triggers task start-up.

        Args:
            task (RLTask): The task to register to the env.
            sim_config: Simulation parameters for physics settings. Defaults to None.
        """

        from omni.isaac.core import SimulationContext
        from omni.isaac.core.world import World

        for key, value in self.config.env.config.task.items():
            if key not in self.config:
                self.config[key] = value

        device = "cpu"
        if sim_config["use_gpu_pipeline"]:
            device = "cuda"

        self._world = World(
            stage_units_in_meters=1.0,
            rendering_dt=1.0 / 30.0,
            backend="torch",
            sim_params=sim_config,
            device=device,
        )
        self._task = task
        self._world.add_task(self._task)
        self.simulation_context = SimulationContext()
        self._world.reset()

    def get_obs_size(self):
        return self._task.get_obs_size()

    def step(self, actions):
        self._task.pre_physics_step(actions)

        self.physics_step()
        self._task.render()

        self._task.post_physics_step()

        obs = self._task.obs_buf
        rew = self._task.rew_buf
        reset = self._task.reset_buf
        extras = self._task.extras

        return obs, rew, reset, extras

    def render(self) -> None:
        """Step the renderer."""
        if self._render:
            self._task.render()
            self._world.render()

    def pre_physics_step(self, actions):
        self._task.pre_physics_step(actions)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._task.reset_envs(env_ids)

        obs = self._task.get_observations()

        return obs

    def reset_world(self):
        """Resets the task and updates observations."""
        # TODO, do we need this?
        self._task.reset()
        self._world.step(render=self._render)
        observations = self._task.get_observations()

        return observations

    def simulate(self):
        self._world.step(render=self._render)

    def close(self) -> None:
        """Closes simulation."""

        # bypass USD warnings on stage close
        self._simulation_app.close()
