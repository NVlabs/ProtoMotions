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

import numpy as np
import torch
from omni.isaac.cloner import GridCloner
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.prims import define_prim
from pxr import Gf

from phys_anim.envs.base_interface.isaacsim.utils.sim_config import SimConfig
from phys_anim.envs.base_interface.isaacsim.utils.usd_utils import (
    create_distant_light,
    create_sphere_light,
)

if TYPE_CHECKING:
    # Just used for autocomplete.
    from phys_anim.envs.humanoid.isaacsim import Humanoid
else:
    Humanoid = object


class SimBaseTask(Humanoid):
    """This class provides a way to set up a task in a scene and modularize adding objects to stage,
    getting observations needed for the behavioral layer, calculating metrics needed about the task,
    calling certain things pre-stepping, creating multiple tasks at the same time and much more.

    Checkout the required tutorials at
    https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html
    """

    def __init__(self, config, device: torch.device) -> None:
        self.config = config
        self.num_envs = config.num_envs
        self.sim_config = self.parse_sim_config()

        self.device = device

        self._scene = None
        self._offset = config.get("offset", None)
        self._task_objects = dict()

        if self._offset is None:
            self._offset = np.array([0.0, 0.0, 0.0])

        # optimization flags for pytorch JIT
        torch._C._jit_set_nvfuser_enabled(False)

        self.control_freq_inv = config.simulator.sim.control_freq_inv

        self._cloner = GridCloner(spacing=config.env_spacing)
        self._cloner.define_base_env(self.default_base_env_path)
        define_prim(self.default_zero_env_path)

    def set_env(self, env):
        self._env = env

    def parse_sim_config(self):
        sim_config = SimConfig(self.config, self.device)

        return sim_config

    @property
    def scene(self) -> Scene:
        """Scene of the world

        Returns:
            Scene: [description]
        """
        return self._scene

    @property
    def name(self) -> str:
        """[summary]

        Returns:
            str: [description]
        """
        return self._name

    def set_up_scene(self, scene: Scene) -> None:
        """Adding assets to the stage as well as adding the encapsulated objects such as XFormPrim..etc
        to the task_objects happens here.

        Args:
            scene (Scene): [description]
        """
        self._scene = scene

        collision_filter_global_paths = list()
        if self.config.sim.get("add_ground_plane", True):
            self._ground_plane_path = "/World/defaultGroundPlane"
            collision_filter_global_paths.append(self._ground_plane_path)
            scene.add_default_ground_plane(prim_path=self._ground_plane_path)
        prim_paths = self._cloner.generate_paths(
            "/World/envs/env", self.config.num_envs
        )
        self._env_pos = self._cloner.clone(
            source_prim_path="/World/envs/env_0",
            prim_paths=prim_paths,
            replicate_physics=True,
        )
        self._env_pos = torch.tensor(
            np.array(self._env_pos), device=self.device, dtype=torch.float
        )
        self._cloner.filter_collisions(
            self._env._world.get_physics_context().prim_path,
            "/World/collisions",
            prim_paths,
            collision_filter_global_paths,
        )
        self.set_initial_camera_params(
            camera_position=[10, 10, 3], camera_target=[0, 0, 0]
        )
        if not self.config.headless:
            light_type = self.config.sim.light_type
            if light_type == "distant":
                create_distant_light()
            elif light_type == "sphere":
                create_sphere_light()
            else:
                raise NotImplementedError(
                    f"Light type {self.config.light_type} is not supported."
                )

    def set_initial_camera_params(
        self, camera_position=[10, 10, 3], camera_target=[0, 0, 0]
    ):
        from omni.kit.viewport.utility import get_viewport_from_window_name
        from omni.kit.viewport.utility.camera_state import ViewportCameraState

        if self._env._render:
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

    def _move_task_objects_to_their_frame(self):
        # if self._task_path:
        # TODO: assumption all task objects are under the same parent
        # Specifying a task path has many limitations atm
        # XFormPrim(prim_path=self._task_path, position=self._offset)
        # for object_name, task_object in self._task_objects.items():
        #     new_prim_path = self._task_path + "/" + task_object.prim_path.split("/")[-1]
        #     task_object.change_prim_path(new_prim_path)
        #     current_position, current_orientation = task_object.get_world_pose()
        for object_name, task_object in self._task_objects.items():
            current_position, current_orientation = task_object.get_world_pose()
            task_object.set_world_pose(position=current_position + self._offset)
            task_object.set_default_state(position=current_position + self._offset)
        return

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

    def get_observations(self) -> dict:
        """Returns current observations from the objects needed for the behavioral layer.

        Raises:
            NotImplementedError: [description]

        Returns:
            dict: [description]
        """
        raise NotImplementedError

    def calculate_metrics(self) -> dict:
        """[summary]

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def is_done(self) -> bool:
        """Returns True of the task is done.

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """called before stepping the physics simulation.

        Args:
            time_step_index (int): [description]
            simulation_time (float): [description]
        """
        return

    def post_reset(self) -> None:
        """Calls while doing a .reset() on the world."""
        return

    def get_description(self) -> str:
        """[summary]

        Returns:
            str: [description]
        """
        return ""

    def cleanup(self) -> None:
        """Called before calling a reset() on the world to removed temporarly objects that were added during
        simulation for instance.
        """
        return

    def set_params(self, *args, **kwargs) -> None:
        """Changes the modifiable paramateres of the task

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def get_params(self) -> dict:
        """Gets the parameters of the task.
        This is defined differently for each task in order to access the task's objects and values.
        Note that this is different from get_observations.
        Things like the robot name, block name..etc can be defined here for faster retrieval.
        should have the form of params_representation["param_name"] = {"value": param_value, "modifiable": bool}

        Raises:
            NotImplementedError: [description]

        Returns:
            dict: defined parameters of the task.
        """
        raise NotImplementedError
