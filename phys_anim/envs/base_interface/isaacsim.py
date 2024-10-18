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

import carb
from omni.isaac.core.world import World
from omni.isaac.cloner import GridCloner
from omni.isaac.core.utils.prims import define_prim
from pxr import Gf, Usd, Sdf, PhysxSchema

from phys_anim.envs.base_interface.common import BaseInterface
from phys_anim.envs.base_interface.isaacsim_utils.sim_config import SimConfig
from phys_anim.envs.base_interface.isaacsim_utils.usd_utils import (
    create_distant_light,
    create_sphere_light,
)

if TYPE_CHECKING:
    # Just used for autocomplete.
    from phys_anim.envs.humanoid.isaacsim import Humanoid
else:
    Humanoid = object


class SimBaseInterface(BaseInterface, Humanoid):
    def __init__(self, config, device: torch.device):
        """
        This class provides a unified interface with IsaacGym environments.
        """

        # IsaacSim does not support substeps.
        # Instead we run at a higher FPS but perform more decimation steps (control_freq_inv).
        # This enables sharing the config between IsaacSim and IsaacGym.
        config.simulator.sim.fps *= config.simulator.sim.substeps
        config.simulator.sim.control_freq_inv *= config.simulator.sim.substeps
        config.simulator.sim.substeps = 1
        config.simulator.sim.dt = 1.0 / config.simulator.sim.fps

        # optimization flags for pytorch JIT
        torch._C._jit_set_nvfuser_enabled(False)

        super().__init__(config, device)

        carb.settings.get_settings().set(
            "/persistent/omnihydra/useSceneGraphInstancing", True
        )

        self.sim_config = self.parse_sim_params()

        self._cloner = GridCloner(spacing=0)
        self._cloner.define_base_env(self.default_base_env_path)
        define_prim(self.default_zero_env_path)

        self._world = World(
            stage_units_in_meters=1.0,
            rendering_dt=1.0 / 30.0,
            backend="torch",
            sim_params=self.sim_config.get_physics_params(),
            device=f"{device.type}:{device.index}",
        )
        self.set_up_scene()

        self._world.reset()

    def on_environment_ready(self):
        self.post_reset()

    @property
    def default_base_env_path(self):
        """Retrieves default path to the parent of all env prims.

        Returns:
            default_base_env_path(str): Defaults to "/World/envs".
        """
        return "/World/envs"

    @property
    def default_zero_env_path(self):
        """Retrieves default path to the first env prim (index 0).

        Returns:
            default_zero_env_path(str): Defaults to "/World/envs/env_0".
        """
        return f"{self.default_base_env_path}/env_0"
    
    @property
    def default_object_path(self):
        return "/World/objects"

    def set_up_scene(self) -> None:
        """Adding assets to the stage as well as adding the encapsulated objects such as XFormPrim..etc
        to the task_objects happens here.
        """

        collision_filter_global_paths = ["/World/terrain"]
        prim_paths = self._cloner.generate_paths(
            "/World/envs/env", self.config.num_envs
        )
        self._cloner.clone(
            source_prim_path="/World/envs/env_0",
            prim_paths=prim_paths,
            replicate_physics=True,
        )
        self._cloner.filter_collisions(
            self._world.get_physics_context().prim_path,
            "/World/collisions",
            prim_paths,
            collision_filter_global_paths,
        )
        self.set_initial_camera_params(
            camera_position=[10, 10, 3], camera_target=[0, 0, 0]
        )
        if not self.config.headless:
            light_type = self.config.simulator.sim.light_type
            if light_type == "distant":
                create_distant_light()
            elif light_type == "sphere":
                create_sphere_light()
            else:
                raise NotImplementedError(
                    f"Light type {self.config.simulator.sim.light_type} is not supported."
                )

    def set_initial_camera_params(
        self, camera_position=[10, 10, 3], camera_target=[0, 0, 0]
    ):
        from omni.kit.viewport.utility import get_viewport_from_window_name
        from omni.kit.viewport.utility.camera_state import ViewportCameraState

        if not self.headless:
            viewport_api_2 = get_viewport_from_window_name("Viewport")
            viewport_api_2.set_active_camera("/OmniverseKit_Persp")
            camera_state = ViewportCameraState("/OmniverseKit_Persp", viewport_api_2)
            camera_state.set_position_world(
                Gf.Vec3d(camera_position[0], camera_position[1], camera_position[2]),
                True,
            )
            camera_state.set_target_world(
                Gf.Vec3d(camera_target[0], camera_target[1], camera_target[2]), True
            )

    def parse_sim_params(self):
        sim_config = SimConfig(self.config.simulator, self.device)
        return sim_config

    def step(self, actions):
        self.pre_physics_step(actions)

        self.physics_step()

        self.render()

        self.post_physics_step()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def simulate(self):
        self._world.step(render=not self.headless)
