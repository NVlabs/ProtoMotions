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

import sys
from isaacgym import gymapi, gymtorch, gymutil  # type: ignore[misc]
import torch
from torch import Tensor
import numpy as np
from rich.progress import Progress
import os
from protomotions.envs.base_env.env_utils.terrains.terrain import Terrain
from protomotions.envs.base_env.env_utils.terrains.flat_terrain import FlatTerrain
from protomotions.utils.scene_lib import SceneLib
from isaac_utils import torch_utils
from typing import Dict, Optional
from protomotions.simulator.base_simulator.robot_state import RobotState
from protomotions.simulator.base_simulator.simulator import Simulator
from protomotions.simulator.base_simulator.config import (
    MarkerState,
    ControlType,
    VisualizationMarker,
    SimBodyOrdering,
    SimulatorConfig
)


class IsaacGymSimulator(Simulator):
    # ===== Group 1: Initialization & Configuration =====
    def __init__(
        self,
        config: SimulatorConfig,
        terrain: Terrain,
        device: torch.device,
        scene_lib: Optional[SceneLib] = None,
        visualization_markers: Optional[Dict[str, VisualizationMarker]] = None,
    ) -> None:
        super().__init__(config=config, scene_lib=scene_lib, terrain=terrain, visualization_markers=visualization_markers, device=device)

        # Handle the case where device.index is None
        if self.device.index is None:
            if self.device.type == 'cuda':
                # Try to get the default CUDA device index
                try:
                    device_index = torch.cuda.current_device()
                except:
                    device_index = 0
            else:
                device_index = 0
        else:
            device_index = self.device.index

        self._graphics_device_id = device_index
        if self.headless is True:
            self._graphics_device_id = -1

        self._gym = gymapi.acquire_gym()

        # create envs, sim and viewer
        self._marker_handles = [[] for _ in range(self.num_envs)]
        self._marker_names_ordering = (
            list(visualization_markers.keys()) if visualization_markers else []
        )
        self._create_sim(visualization_markers)
        self._gym.prepare_sim(self._sim)
        self._enable_viewer_sync = True
        self._viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if not self.headless:
            # subscribe to keyboard shortcuts
            self._viewer = self._gym.create_viewer(self._sim, gymapi.CameraProperties())
            self._gym.subscribe_viewer_keyboard_event(self._viewer, gymapi.KEY_Q, "QUIT")
            self._gym.subscribe_viewer_keyboard_event(
                self._viewer, gymapi.KEY_V, "toggle_viewer_sync"
            )
            self._gym.subscribe_viewer_keyboard_event(
                self._viewer, gymapi.KEY_J, "push_robot"
            )
            self._gym.subscribe_viewer_keyboard_event(
                self._viewer, gymapi.KEY_L, "toggle_video_record"
            )
            self._gym.subscribe_viewer_keyboard_event(
                self._viewer, gymapi.KEY_SEMICOLON, "cancel_video_record"
            )
            self._gym.subscribe_viewer_keyboard_event(
                self._viewer, gymapi.KEY_R, "reset_envs"
            )
            self._gym.subscribe_viewer_keyboard_event(
                self._viewer, gymapi.KEY_O, "toggle_camera_target"
            )

            # set the camera position based on up axis
            sim_params = self._gym.get_sim_params(self._sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
                cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            self._gym.viewer_camera_look_at(self._viewer, None, cam_pos, cam_target)

        # Refresh tensors BEFORE we acquire them https://forums.developer.nvidia.com/t/isaacgym-preview-4-actor-root-state-returns-nans-with-isaacgymenvs-style-task/223738/4
        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)
        self._gym.refresh_net_contact_force_tensor(self._sim)
        self._gym.refresh_force_sensor_tensor(self._sim)

        # get gym GPU state tensors
        actor_root_state = self._gym.acquire_actor_root_state_tensor(self._sim)
        dof_state_tensor = self._gym.acquire_dof_state_tensor(self._sim)
        rigid_body_state = self._gym.acquire_rigid_body_state_tensor(self._sim)
        contact_force_tensor = self._gym.acquire_net_contact_force_tensor(self._sim)

        dof_force_tensor = self._gym.acquire_dof_force_tensor(self._sim)
        self._dof_force_tensor: Tensor = gymtorch.wrap_tensor(dof_force_tensor).view(
            self.num_envs, self._num_dof
        )

        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)
        self._gym.refresh_net_contact_force_tensor(self._sim)
        self._gym.refresh_force_sensor_tensor(self._sim)

        self._root_states: Tensor = gymtorch.wrap_tensor(actor_root_state)

        num_actors = self._get_num_actors_per_env()

        self._humanoid_root_states = self._root_states.view(
            self.num_envs, num_actors, actor_root_state.shape[-1]
        )[..., 0, :]

        self._object_root_states = self._root_states.view(
            self.num_envs, num_actors, actor_root_state.shape[-1]
        )[..., 1: self._num_objects_per_scene + 1, :]
        self._object_indices = torch_utils.to_torch(
            self._object_indices, dtype=torch.int32, device=self.device
        )

        self._humanoid_actor_ids = num_actors * torch.arange(
            self.num_envs, device=self.device, dtype=torch.int32
        )

        # create some wrapper tensors for different slices
        self._dof_state: Tensor = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        self._dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[
            ..., : self._num_dof, 0
        ]
        self._dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[
            ..., : self._num_dof, 1
        ]

        self._rigid_body_state: Tensor = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        rigid_body_state_reshaped = self._rigid_body_state.view(
            self.num_envs, bodies_per_env, 13
        )

        self._rigid_body_pos = rigid_body_state_reshaped[..., : self._num_bodies, 0:3]
        self._rigid_body_rot = rigid_body_state_reshaped[..., : self._num_bodies, 3:7]
        self._rigid_body_vel = rigid_body_state_reshaped[..., : self._num_bodies, 7:10]
        self._rigid_body_ang_vel = rigid_body_state_reshaped[
            ..., : self._num_bodies, 10:13
        ]

        self._reset_states = RobotState(
            root_pos=torch.zeros(
                self.num_envs, 3, dtype=torch.float, device=self.device
            ),
            root_rot=torch.zeros(
                self.num_envs, 4, dtype=torch.float, device=self.device
            ),
            root_vel=torch.zeros(
                self.num_envs, 3, dtype=torch.float, device=self.device
            ),
            root_ang_vel=torch.zeros(
                self.num_envs, 3, dtype=torch.float, device=self.device
            ),
            dof_pos=torch.zeros(
                self.num_envs, self._num_dof, dtype=torch.float, device=self.device
            ),
            dof_vel=torch.zeros(
                self.num_envs, self._num_dof, dtype=torch.float, device=self.device
            ),
            rigid_body_pos=torch.zeros(
                self.num_envs, self._num_bodies, 3, dtype=torch.float, device=self.device
            ),
            rigid_body_rot=torch.zeros(
                self.num_envs, self._num_bodies, 4, dtype=torch.float, device=self.device
            ),
            rigid_body_vel=torch.zeros(
                self.num_envs, self._num_bodies, 3, dtype=torch.float, device=self.device
            ),
            rigid_body_ang_vel=torch.zeros(
                self.num_envs, self._num_bodies, 3, dtype=torch.float, device=self.device
            ),
        )

        initial_humanoid_root_states = self._humanoid_root_states.clone()
        initial_humanoid_root_states[:, 7:13] = 0
        self._isaacgym_default_state = RobotState(
            root_pos=initial_humanoid_root_states[:, :3],
            root_rot=initial_humanoid_root_states[:, 3:7],
            root_vel=initial_humanoid_root_states[:, 7:10],
            root_ang_vel=initial_humanoid_root_states[:, 10:13],
            dof_pos=self._dof_pos.clone(),
            dof_vel=self._dof_vel.clone(),
            rigid_body_pos=self._rigid_body_pos.clone(),
            rigid_body_rot=self._rigid_body_rot.clone(),
            rigid_body_vel=self._rigid_body_vel.clone(),
            rigid_body_ang_vel=self._rigid_body_ang_vel.clone(),
        )

        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces = contact_force_tensor.view(
            self.num_envs, bodies_per_env, 3
        )[..., : self._num_bodies, :]

        if self.scene_lib is None or self.scene_lib.num_objects_per_scene == 0:
            self._object_contact_forces = None
        else:
            self._object_contact_forces = contact_force_tensor.view(
                self.num_envs, bodies_per_env, 3
            )[..., self._num_bodies :, :]

        if not self.headless:
            self._init_camera()

        if visualization_markers:
            self._build_marker_state_tensors()

    # ===== Group 2: Environment Setup & Configuration =====
    def _load_marker_asset(self) -> None:
        asset_root = "protomotions/data/assets/urdf/"
        asset_file = "traj_marker.urdf"
        small_asset_file = "traj_marker_small.urdf"
        tiny_asset_file = "traj_marker_tiny.urdf"
        arrow_asset_file = "heading_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self._gym.load_asset(
            self._sim, asset_root, asset_file, asset_options
        )
        self._marker_asset_small = self._gym.load_asset(
            self._sim, asset_root, small_asset_file, asset_options
        )
        self._marker_asset_tiny = self._gym.load_asset(
            self._sim, asset_root, tiny_asset_file, asset_options
        )
        self._marker_asset_arrow = self._gym.load_asset(
            self._sim, asset_root, arrow_asset_file, asset_options
        )

    def _set_sim_params_up_axis(self, sim_params: gymapi.SimParams, axis: str) -> int:
        if axis == "z":
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity.x = 0
            sim_params.gravity.y = 0
            sim_params.gravity.z = -9.81
            return 2
        return 1

    def _add_terrain(self) -> None:
        print("Adding terrain")
        if isinstance(self.terrain, FlatTerrain):
            # configure the ground plane
            plane_params = gymapi.PlaneParams()
            plane_params.normal = gymapi.Vec3(0, 0, 1)
            plane_params.distance = 0
            plane_params.static_friction = self.config.plane.static_friction
            plane_params.dynamic_friction = self.config.plane.dynamic_friction
            plane_params.restitution = self.config.plane.restitution

            # create the ground plane
            self._gym.add_ground(self._sim, plane_params)
        else:
            tm_params = gymapi.TriangleMeshParams()
            tm_params.nb_vertices = self.terrain.vertices.shape[0]
            tm_params.nb_triangles = self.terrain.triangles.shape[0]
            tm_params.transform.p.x = 0
            tm_params.transform.p.y = 0
            tm_params.transform.p.z = 0.0
            tm_params.static_friction = self.config.plane.static_friction
            tm_params.dynamic_friction = self.config.plane.dynamic_friction
            tm_params.restitution = self.config.plane.restitution
            self._gym.add_triangle_mesh(
                self._sim,
                self.terrain.vertices.flatten(order="C"),
                self.terrain.triangles.flatten(order="C"),
                tm_params,
            )
        print("Terrain added")

    def _parse_sim_params(self) -> gymapi.SimParams:
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / self.config.sim.fps
        sim_params.num_client_threads = 0

        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = True
        sim_params.physx.num_subscenes = 0
        sim_params.physx.max_gpu_contact_pairs = (
            self.robot_config.contact_pairs_multiplier * 1024 * 1024
        )
        sim_params.use_gpu_pipeline = True

        gymutil.parse_sim_config(self.config.sim, sim_params)
        return sim_params

    def _create_sim(self, visualization_markers: Optional[Dict[str, VisualizationMarker]] = None) -> None:
        self._sim_params = self._parse_sim_params()
        self._physics_engine = gymapi.SIM_PHYSX

        self._plane_static_friction = self.config.plane.static_friction
        self._plane_dynamic_friction = self.config.plane.dynamic_friction
        self._plane_restitution = self.config.plane.restitution

        self._up_axis_idx = self._set_sim_params_up_axis(self._sim_params, "z")

        sim = self._gym.create_sim(
            self._graphics_device_id if self.device.index is None else self.device.index,
            self._graphics_device_id,
            self._physics_engine,
            self._sim_params,
        )
        if sim is None:
            print("*** Failed to create sim")
            quit()

        self._sim = sim

        self._add_terrain()
        self._load_marker_asset()

        self._create_envs(
            0,
            int(np.sqrt(self.num_envs)),
            visualization_markers,
        )

    def _create_envs(
        self, spacing: float, num_per_row: int, visualization_markers: Optional[Dict[str, VisualizationMarker]] = None
    ) -> None:
        lower = gymapi.Vec3(0.0, 0.0, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = self.robot_config.asset.asset_root
        asset_file = self.robot_config.asset.asset_file_name

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        def set_value_if_not_none(prev_value, new_value):
            return new_value if new_value is not None else prev_value

        asset_config_options = [
            "collapse_fixed_joints",
            "replace_cylinder_with_capsule",
            "flip_visual_attachments",
            "armature",
            "thickness",
            "max_angular_velocity",
            "max_linear_velocity",
            "density",
            "angular_damping",
            "linear_damping",
            "disable_gravity",
            "fix_base_link",
            "default_dof_drive_mode",
        ]
        for option in asset_config_options:
            option_value = set_value_if_not_none(
                getattr(asset_options, option), getattr(self.robot_config.asset, option)
            )
            setattr(asset_options, option, option_value)

        self._humanoid_asset = humanoid_asset = self._gym.load_asset(
            self._sim, asset_root, asset_file, asset_options
        )

        robot_num_bodies = self._gym.get_asset_rigid_body_count(humanoid_asset)
        assert (
            robot_num_bodies == self.robot_config.num_bodies
        ), f"Number of bodies in the config {self.robot_config.num_bodies} doesn't match provided robot {robot_num_bodies}"
        self._dof_names = self._gym.get_asset_dof_names(humanoid_asset)
        robot_num_dof = self._gym.get_asset_dof_count(humanoid_asset)
        assert robot_num_dof == len(
            self.robot_config.dof_names
        ), f"Number of dofs in the config {len(self.robot_config.dof_names)} doesn't match provided robot {robot_num_dof}"
        self._num_joints = self._gym.get_asset_joint_count(humanoid_asset)

        self._humanoid_handles = []
        self._object_handles = []
        self._object_indices = []
        self._envs = []
        dof_limits_lower = []
        dof_limits_upper = []

        self._object_assets = []
        if self.scene_lib is not None and self.scene_lib.total_spawned_scenes > 0:
            self._load_object_assets()

        with Progress() as progress:
            task = progress.add_task(
                f"[cyan]Creating {self.num_envs} environments...", total=self.num_envs
            )
            for i in range(self.num_envs):
                env_ptr = self._gym.create_env(self._sim, lower, upper, num_per_row)
                self._build_env(i, env_ptr, humanoid_asset, visualization_markers)
                self._envs.append(env_ptr)
                progress.update(task, advance=1)

        dof_prop = self._gym.get_actor_dof_properties(
            self._envs[0], self._humanoid_handles[0]
        )
        for j in range(len(dof_prop["upper"])):
            if dof_prop["lower"][j] > dof_prop["upper"][j]:
                dof_limits_lower.append(dof_prop["upper"][j])
                dof_limits_upper.append(dof_prop["lower"][j])
            else:
                dof_limits_lower.append(dof_prop["lower"][j])
                dof_limits_upper.append(dof_prop["upper"][j])

        self._dof_limits_lower_sim = torch_utils.to_torch(
            dof_limits_lower, device=self.device
        )
        self._dof_limits_upper_sim = torch_utils.to_torch(
            dof_limits_upper, device=self.device
        )

    def _load_object_assets(self) -> None:
        if self.scene_lib.total_spawned_scenes > 0:
            self._object_names = []

            with Progress() as progress:
                task = progress.add_task(
                    "[green]Loading object assets...",
                    total=len(self.scene_lib.object_spawn_list),
                )

                for i, object_info in enumerate(self.scene_lib.object_spawn_list):
                    object_name = os.path.splitext(
                        os.path.basename(object_info.object_path)
                    )[0]

                    if object_info.is_first_instance:
                        object_options_dict = object_info.object_options.to_dict()
                        object_asset_options = gymapi.AssetOptions()
                        if object_options_dict.get("vhacd_enabled", False):
                            object_asset_options.vhacd_params = gymapi.VhacdParams()
                        for key, value in object_options_dict.items():
                            if type(value) is dict:
                                if hasattr(object_asset_options, key):
                                    object_asset_sub_options = getattr(
                                        object_asset_options, key
                                    )
                                    for sub_key, sub_value in value.items():
                                        if hasattr(object_asset_sub_options, sub_key):
                                            setattr(
                                                object_asset_sub_options,
                                                sub_key,
                                                sub_value,
                                            )
                                else:
                                    print(
                                        f"Warning: {key} is not a valid option for object asset"
                                    )
                            else:
                                if hasattr(object_asset_options, key):
                                    if key == "default_dof_drive_mode":
                                        value = getattr(gymapi, value)
                                    setattr(object_asset_options, key, value)
                                else:
                                    print(
                                        f"Warning: {key} is not a valid option for object asset"
                                    )
                        object_asset = self._gym.load_asset(
                            self._sim,
                            os.path.dirname(object_info.object_path),
                            f"{object_name}.urdf",
                            object_asset_options,
                        )
                        sensor_pose = gymapi.Transform()
                        sensor_options = gymapi.ForceSensorProperties()
                        sensor_options.enable_forward_dynamics_forces = False
                        sensor_options.enable_constraint_solver_forces = True
                        sensor_options.use_world_frame = False
                        self._gym.create_asset_force_sensor(
                            object_asset, 0, sensor_pose, sensor_options
                        )
                    else:
                        object_asset = self._object_assets[object_info.first_instance_id]

                    self._object_assets.append(object_asset)
                    self._object_names.append(object_name)
                    progress.update(task, advance=1)

            print(
                f"=========== Total number of unique objects is {len(self._object_assets)}"
            )

    def _build_env(
        self, env_id: int, env_ptr, humanoid_asset, visualization_markers: Optional[Dict[str, VisualizationMarker]] = None
    ) -> None:
        col_group = env_id
        col_filter = 0 if self.robot_config.asset.self_collisions else 1
        segmentation_id = 0

        start_pose = gymapi.Transform()
        start_offset = [env_id, env_id, env_id]
        start_pose.p = gymapi.Vec3(*start_offset)
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        humanoid_handle = self._gym.create_actor(
            env_ptr,
            humanoid_asset,
            start_pose,
            "humanoid",
            col_group,
            col_filter,
            segmentation_id,
        )

        self._gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

        for j in range(self.robot_config.num_bodies):
            self._gym.set_rigid_body_color(
                env_ptr,
                humanoid_handle,
                j,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(0.54, 0.85, 0.2),
            )

        dof_prop = self._gym.get_asset_dof_properties(humanoid_asset)
        if self.control_type == ControlType.BUILT_IN_PD:
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
        else:
            dof_prop["driveMode"] = gymapi.DOF_MODE_EFFORT

        self._gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)

        filter_ints = self.robot_config.asset.filter_ints
        if filter_ints is not None:
            props = self._gym.get_actor_rigid_shape_properties(env_ptr, humanoid_handle)
            assert len(filter_ints) == len(props)
            for p_idx in range(len(props)):
                props[p_idx].filter = filter_ints[p_idx]
            self._gym.set_actor_rigid_shape_properties(env_ptr, humanoid_handle, props)

        self._humanoid_handles.append(humanoid_handle)

        if self.scene_lib is not None and self.scene_lib.num_objects_per_scene > 0:
            self._build_object_playground(env_id, env_ptr)

        self._build_markers(env_id, env_ptr, visualization_markers)

    def _build_object_playground(self, env_id: int, env_ptr) -> None:
        print(f"=========== Building object playground for env {env_id}")
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        scene_spawn_info = self.scene_lib.scenes[env_id]
        scene_offset = self.scene_lib.scene_offsets[env_id]

        height_at_scene_origin = self.terrain.get_ground_heights(
            torch.tensor(
                [[scene_offset[0], scene_offset[1]]],
                device=self.device,
                dtype=torch.float,
            )
        ).item()
        self._scene_position.append(
            torch.tensor(
                [scene_offset[0], scene_offset[1], height_at_scene_origin],
                device=self.device,
                dtype=torch.float,
            )
        )
        self._object_dims.append([])

        for obj in scene_spawn_info.objects:
            # Get the spawn info for this object which contains the correct ID
            object_spawn_info = next(
                info
                for info in self.scene_lib.object_spawn_list
                if info.object_path == obj.object_path
                and (info.is_first_instance or info.first_instance_id == info.id)
            )
            object_id = object_spawn_info.id

            object_asset = self._object_assets[object_id]
            object_name = object_spawn_info.object_path.split("/")[-1].split(".")[0]
            object_pose = gymapi.Transform()

            global_object_position = torch.tensor(
                [
                    scene_offset[0] + obj.translation[0],
                    scene_offset[1] + obj.translation[1],
                    0 + obj.translation[2],
                ],
                device=self.device,
                dtype=torch.float,
            )

            object_pose.p = gymapi.Vec3(
                global_object_position[0],
                global_object_position[1],
                global_object_position[2],
            )
            object_pose.r = gymapi.Quat(
                obj.rotation[0],
                obj.rotation[1],
                obj.rotation[2],
                obj.rotation[3],
            )

            object_handle = self._gym.create_actor(
                env_ptr,
                object_asset,
                object_pose,
                object_name,
                col_group,
                col_filter,
                segmentation_id,
            )
            self._object_handles.append(object_handle)
            object_idx = self._gym.get_actor_index(
                env_ptr, object_handle, gymapi.DOMAIN_SIM
            )
            self._object_indices.append(object_idx)
            object_dims = torch.tensor(object_spawn_info.object_dims, device=self.device, dtype=torch.float)
            self._object_dims[-1].append(object_dims)

        self._object_dims[-1] = torch.stack(self._object_dims[-1]).reshape(
            self._num_objects_per_scene, -1
        )

    def _build_markers(self, env_id: int, env_ptr, visualization_markers: Optional[Dict[str, VisualizationMarker]] = None) -> None:
        """Build visualization markers for the environment.

        Args:
            env_id (int): Environment ID to build markers for
            env_ptr: Environment pointer from IsaacGym
            visualization_markers (Dict[str, VisualizationMarker]): Dictionary mapping marker names to their configurations
        """
        if visualization_markers is None:
            return

        for marker_name, markers_cfg in visualization_markers.items():
            for i, marker in enumerate(markers_cfg.markers):
                if markers_cfg.type == "sphere":
                    if marker.size == "tiny":
                        marker_asset = self._marker_asset_tiny
                    elif marker.size == "small":
                        marker_asset = self._marker_asset_small
                    else:
                        marker_asset = self._marker_asset
                elif markers_cfg.type == "arrow":
                    if marker.size == "regular":
                        marker_asset = self._marker_asset_arrow
                    else:
                        raise ValueError(
                            f"Marker size {marker.size} not supported for arrow markers in IsaacGym"
                        )
                else:
                    raise ValueError(f"Marker type {markers_cfg.type} not supported")

                marker_handle = self._gym.create_actor(
                    env_ptr,
                    marker_asset,
                    gymapi.Transform(),
                    marker_name,
                    self.num_envs + 10,  # A unique collision group for markers
                    0,
                    0,
                )
                color = gymapi.Vec3(
                    markers_cfg.color[0], markers_cfg.color[1], markers_cfg.color[2]
                )
                self._gym.set_rigid_body_color(
                    env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, color
                )
                self._marker_handles[env_id].append(marker_handle)

    def _build_marker_state_tensors(self):
        num_actors = self._get_num_actors_per_env()
        if self._num_objects_per_scene > 0:
            self._marker_states = self._root_states.view(
                self.num_envs, num_actors, self._root_states.shape[-1]
            )[..., 1 + self._num_objects_per_scene:, :]
        else:
            self._marker_states = self._root_states.view(
                self.num_envs, num_actors, self._root_states.shape[-1]
            )[..., 1:, :]
        self._marker_pos = self._marker_states[..., :3]
        self._marker_rot = self._marker_states[..., 3:7]

        self._marker_actor_ids = self._humanoid_actor_ids.unsqueeze(
            -1
        ) + torch_utils.to_torch(
            self._marker_handles, dtype=torch.int32, device=self.device
        )
        self._marker_actor_ids = self._marker_actor_ids.flatten()

    # ===== Group 3: Simulation Steps & State Management =====
    def _physics_step(self) -> None:
        if self.control_type == ControlType.BUILT_IN_PD:
            self._apply_pd_control()
        for i in range(self.decimation):
            if not self.control_type == ControlType.BUILT_IN_PD:
                self._apply_motor_forces()
            self._simulate()
            if self.device.type == "cpu":
                self._gym.fetch_results(self._sim, True)
            if not self.control_type == ControlType.BUILT_IN_PD:
                self._gym.refresh_dof_state_tensor(self._sim)
        self._refresh_sim_tensors()

    def _refresh_sim_tensors(self) -> None:
        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)

        if len(self.reset_env_ids) > 0:
            env_ids = self.reset_env_ids
            self._humanoid_root_states[env_ids, 0:3] = self._reset_states.root_pos[
                env_ids
            ]
            self._humanoid_root_states[env_ids, 3:7] = self._reset_states.root_rot[
                env_ids
            ]
            self._humanoid_root_states[env_ids, 7:10] = self._reset_states.root_vel[
                env_ids
            ]
            self._humanoid_root_states[env_ids, 10:13] = self._reset_states.root_ang_vel[
                env_ids
            ]

            self._dof_pos[env_ids] = self._reset_states.dof_pos[env_ids]
            self._dof_vel[env_ids] = self._reset_states.dof_vel[env_ids]

            self._rigid_body_pos[env_ids] = self._reset_states.rigid_body_pos[env_ids]
            self._rigid_body_rot[env_ids] = self._reset_states.rigid_body_rot[env_ids]
            self._rigid_body_vel[env_ids] = self._reset_states.rigid_body_vel[env_ids]
            self._rigid_body_ang_vel[env_ids] = self._reset_states.rigid_body_ang_vel[
                env_ids
            ]
            self.reset_env_ids = []

        self._gym.refresh_force_sensor_tensor(self._sim)
        self._gym.refresh_dof_force_tensor(self._sim)
        self._gym.refresh_net_contact_force_tensor(self._sim)

    def _update_simulator_tensors_after_reset(
        self, env_ids: Optional[torch.Tensor]
    ) -> None:
        actor_ids = self._humanoid_actor_ids[env_ids]
        set_root_state_ids = actor_ids

        self._gym.set_actor_root_state_tensor_indexed(
            self._sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(set_root_state_ids),
            len(set_root_state_ids),
        )
        self._gym.set_dof_state_tensor_indexed(
            self._sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids),
        )
        self._gym.set_dof_position_target_tensor_indexed(
            self._sim,
            gymtorch.unwrap_tensor(self._dof_pos.contiguous()),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids),
        )
        self._refresh_sim_tensors()

    def _set_simulator_env_state(
        self, new_states: RobotState, env_ids: Optional[torch.Tensor]
    ) -> None:
        self._humanoid_root_states[env_ids, 0:3] = new_states.root_pos
        self._humanoid_root_states[env_ids, 3:7] = new_states.root_rot
        self._humanoid_root_states[env_ids, 7:10] = new_states.root_vel
        self._humanoid_root_states[env_ids, 10:13] = new_states.root_ang_vel

        self._dof_pos[env_ids] = new_states.dof_pos
        self._dof_vel[env_ids] = new_states.dof_vel

        self._rigid_body_pos[env_ids] = new_states.rigid_body_pos
        self._rigid_body_rot[env_ids] = new_states.rigid_body_rot
        self._rigid_body_vel[env_ids] = new_states.rigid_body_vel
        self._rigid_body_ang_vel[env_ids] = new_states.rigid_body_ang_vel

        self._reset_states.root_pos[env_ids] = new_states.root_pos.clone()
        self._reset_states.root_rot[env_ids] = new_states.root_rot.clone()
        self._reset_states.root_vel[env_ids] = new_states.root_vel.clone()
        self._reset_states.root_ang_vel[env_ids] = new_states.root_ang_vel.clone()
        self._reset_states.dof_pos[env_ids] = new_states.dof_pos.clone()
        self._reset_states.dof_vel[env_ids] = new_states.dof_vel.clone()
        self._reset_states.rigid_body_pos[env_ids] = new_states.rigid_body_pos.clone()
        self._reset_states.rigid_body_rot[env_ids] = new_states.rigid_body_rot.clone()
        self._reset_states.rigid_body_vel[env_ids] = new_states.rigid_body_vel.clone()
        self._reset_states.rigid_body_ang_vel[env_ids] = (
            new_states.rigid_body_ang_vel.clone()
        )
        self.reset_env_ids = env_ids

    def _simulate(self) -> None:
        self._gym.simulate(self._sim)
        if self.device == "cpu":
            self._gym.fetch_results(self._sim, True)
        self._gym.refresh_dof_state_tensor(self._sim)

    # ===== Group 4: State Getters =====
    def _get_simulator_default_state(self) -> RobotState:
        return self._isaacgym_default_state

    def _get_num_actors_per_env(self) -> int:
        num_actors = self._root_states.shape[0] // self.num_envs
        return num_actors

    def _get_sim_body_ordering(self) -> SimBodyOrdering:
        return SimBodyOrdering(
            body_names=self._gym.get_asset_rigid_body_names(self._humanoid_asset),
            dof_names=self._gym.get_asset_dof_names(self._humanoid_asset),
            contact_sensor_body_names=self._gym.get_asset_rigid_body_names(
                self._humanoid_asset
            ),
        )

    def _get_simulator_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        root_pos = self._humanoid_root_states[..., :3].clone()
        root_rot = self._humanoid_root_states[..., 3:7].clone()
        root_vel = self._humanoid_root_states[..., 7:10].clone()
        root_ang_vel = self._humanoid_root_states[..., 10:13].clone()
        if env_ids is not None:
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            root_vel = root_vel[env_ids]
            root_ang_vel = root_ang_vel[env_ids]
        return RobotState(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
        )

    def _get_simulator_object_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        root_pos = self._object_root_states[..., :3].clone()
        root_rot = self._object_root_states[..., 3:7].clone()
        root_vel = self._object_root_states[..., 7:10].clone()
        root_ang_vel = self._object_root_states[..., 10:13].clone()
        if env_ids is not None:
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            root_vel = root_vel[env_ids]
            root_ang_vel = root_ang_vel[env_ids]
        return RobotState(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
        )

    def _get_simulator_bodies_contact_buf(self, env_ids=None):
        contact_forces = self._contact_forces.clone()
        if env_ids is not None:
            contact_forces = contact_forces[env_ids]
        return contact_forces

    def _get_simulator_object_contact_buf(self, env_ids=None):
        object_contact_forces = self._object_contact_forces.clone()
        if env_ids is not None:
            object_contact_forces = object_contact_forces[env_ids]
        return object_contact_forces

    def _get_simulator_bodies_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        body_pos = self._rigid_body_pos.clone()
        body_rot = self._rigid_body_rot.clone()
        body_vel = self._rigid_body_vel.clone()
        body_ang_vel = self._rigid_body_ang_vel.clone()
        if env_ids is not None:
            body_pos = body_pos[env_ids]
            body_rot = body_rot[env_ids]
            body_vel = body_vel[env_ids]
            body_ang_vel = body_ang_vel[env_ids]
        return RobotState(
            rigid_body_pos=body_pos,
            rigid_body_rot=body_rot,
            rigid_body_vel=body_vel,
            rigid_body_ang_vel=body_ang_vel,
        )

    def _get_simulator_dof_forces(self, env_ids=None):
        dof_forces = self._dof_force_tensor.clone()
        if env_ids is not None:
            dof_forces = dof_forces[env_ids]
        return dof_forces

    def _get_simulator_dof_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        dof_pos = self._dof_pos.clone()
        dof_vel = self._dof_vel.clone()
        if env_ids is not None:
            dof_pos = dof_pos[env_ids]
            dof_vel = dof_vel[env_ids]
        return RobotState(
            dof_pos=dof_pos,
            dof_vel=dof_vel,
        )

    # ===== Group 5: Control & Computation Methods =====
    def _apply_pd_control(self) -> None:
        pd_tar = self._action_to_pd_targets(self._actions)
        pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
        self._gym.set_dof_position_target_tensor(self._sim, pd_tar_tensor)

    def _apply_motor_forces(self) -> None:
        torques = self._compute_torques(self._actions)
        torques_tensor = gymtorch.unwrap_tensor(torques)
        self._gym.set_dof_actuation_force_tensor(self._sim, torques_tensor)
        
    def _push_robot(self):
        forces = torch.zeros((1, self._rigid_body_state.shape[0], 3), device=self.device, dtype=torch.float)
        torques = torch.zeros((1, self._rigid_body_state.shape[0], 3), device=self.device, dtype=torch.float)
        
        # Apply force to the pelvis
        for i in range(self._rigid_body_state.shape[0] // self._num_bodies):
            forces[:, i * self._num_bodies, :] = -8000
            forces[:, i * self._num_bodies, :] = -8000

        self._gym.apply_rigid_body_force_tensors(self._sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)

    # ===== Group 6: Rendering & Visualization =====
    def _init_camera(self) -> None:
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._cam_prev_char_pos = (
            self._get_simulator_root_state(self._camera_target["env"])
            .root_pos.cpu()
            .numpy()
        )

        cam_pos = gymapi.Vec3(
            self._cam_prev_char_pos[0],
            self._cam_prev_char_pos[1] - 3.0,
            self._cam_prev_char_pos[2] + 0.4,
        )
        cam_target = gymapi.Vec3(
            self._cam_prev_char_pos[0],
            self._cam_prev_char_pos[1],
            self._cam_prev_char_pos[2] + 0.2,
        )
        self._gym.viewer_camera_look_at(self._viewer, None, cam_pos, cam_target)

    def render(self) -> None:
        if not self.headless:
            self._update_camera()

            # check for window closed
            if self._gym.query_viewer_has_closed(self._viewer):
                sys.exit()

            # check for keyboard events
            for evt in self._gym.query_viewer_action_events(self._viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self._enable_viewer_sync = not self._enable_viewer_sync
                elif evt.action == "push_robot" and evt.value > 0:
                    self._push_robot()
                elif evt.action == "toggle_video_record" and evt.value > 0:
                    self._toggle_video_record()
                elif evt.action == "cancel_video_record" and evt.value > 0:
                    self._cancel_video_record()
                elif evt.action == "reset_envs" and evt.value > 0:
                    self._requested_reset()
                elif evt.action == "toggle_camera_target" and evt.value > 0:
                    self._toggle_camera_target()

            if self.device.type != "cpu":
                self._gym.fetch_results(self._sim, True)

            if self._enable_viewer_sync:
                self._gym.step_graphics(self._sim)
                self._gym.draw_viewer(self._viewer, self._sim, True)
            else:
                self._gym.poll_viewer_events(self._viewer)
        super().render()

    def _update_simulator_markers(self, markers_state: Optional[Dict[str, MarkerState]] = None) -> None:
        if markers_state is None:
            return

        markers_translations = torch.cat(
            [
                markers_state[marker_key].translation
                for marker_key in self._marker_names_ordering
            ],
            dim=1,
        )
        self._marker_pos[:] = markers_translations
        markers_orientations = torch.cat(
            [
                markers_state[marker_key].orientation
                for marker_key in self._marker_names_ordering
            ],
            dim=1,
        )
        self._marker_rot[:] = markers_orientations
        self._gym.set_actor_root_state_tensor_indexed(
            self._sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(self._marker_actor_ids),
            len(self._marker_actor_ids),
        )

    def _write_viewport_to_file(self, file_name: str) -> None:
        self._gym.write_viewer_image_to_file(
            self._viewer,
            file_name,
        )

    def close(self) -> None:
        if self._viewer:
            self._gym.destroy_viewer(self._viewer)
        self._gym.destroy_sim(self._sim)

    def _update_camera(self) -> None:
        self._gym.refresh_actor_root_state_tensor(self._sim)

        if self._camera_target["element"] == 0:
            current_char_pos = (
                self._get_simulator_root_state(self._camera_target["env"])
                .root_pos.cpu()
                .numpy()
            )
            height_offset = 0.2
        else:
            in_scene_object_id = self._camera_target["element"] - 1
            current_char_pos = (
                self._get_simulator_object_root_state(self._camera_target["env"])
                .root_pos[in_scene_object_id]
                .cpu()
                .numpy()
            )
            height_offset = 0

        current_cam_transform = self._gym.get_viewer_camera_transform(self._viewer, None)
        current_cam_pos = np.array(
            [
                current_cam_transform.p.x,
                current_cam_transform.p.y,
                current_cam_transform.p.z,
            ]
        )

        cam_offset = current_cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(
            current_char_pos[0],
            current_char_pos[1],
            current_char_pos[2] + height_offset,
        )

        new_cam_pos = gymapi.Vec3(
            current_char_pos[0] + cam_offset[0],
            current_char_pos[1] + cam_offset[1],
            current_char_pos[2] + cam_offset[2],
        )

        self._gym.viewer_camera_look_at(self._viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = current_char_pos
