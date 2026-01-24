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
import sys
from dataclasses import asdict
from isaacgym import gymapi, gymtorch, gymutil  # type: ignore[misc]
import torch
from torch import Tensor
import numpy as np
from rich.progress import Progress
import os
from protomotions.components.terrains.terrain import Terrain
from protomotions.components.terrains.config import CombineMode
from protomotions.components.scene_lib import (
    SceneLib,
    PrimitiveSceneObject,
    SceneObject,
    BoxSceneObject,
    SphereSceneObject,
    CylinderSceneObject,
    MeshSceneObject,
)
from protomotions.utils import torch_utils
from typing import Dict, Optional, List, Tuple
from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    RootOnlyState,
    StateConversion,
    ObjectState,
    ResetState,
)
from protomotions.simulator.base_simulator.simulator import Simulator, ControlType
from protomotions.simulator.base_simulator.config import (
    MarkerState,
    VisualizationMarkerConfig,
    SimBodyOrdering,
    SimulatorConfig,
)
import tempfile


class IsaacGymSimulator(Simulator):
    # ===== Group 1: Initialization & Configuration =====
    def __init__(
        self,
        config: SimulatorConfig,
        robot_config,
        terrain: Optional[Terrain],
        device: torch.device,
        scene_lib: SceneLib,
        custom_key_handlers: Optional[Dict[str, callable]] = None,
    ) -> None:
        super().__init__(
            config=config,
            robot_config=robot_config,
            scene_lib=scene_lib,
            terrain=terrain,
            device=device,
        )

        # Store custom key handlers
        self._custom_key_handlers = custom_key_handlers or {}

        # Create temporary directory for primitive URDFs
        self._temp_dir = tempfile.TemporaryDirectory()

        # Handle the case where device.index is None
        if self.device.index is None:
            if self.device.type == "cuda":
                # Try to get the default CUDA device index
                try:
                    device_index = torch.cuda.current_device()
                except Exception:
                    device_index = 0
            else:
                device_index = 0
        else:
            device_index = self.device.index

        self._graphics_device_id = device_index
        if self.headless is True:
            self._graphics_device_id = 0

        self._gym = gymapi.acquire_gym()

        # Prepare for marker setup (will be populated in _create_simulation)
        self._marker_handles = [[] for _ in range(self.num_envs)]
        self._marker_names_ordering = []

        # Texture cache: maps texture file paths to gymapi texture handles
        self._texture_handles = {}

    def _create_simulation(self) -> None:
        """Create the IsaacGym simulation environment.

        Called by base class _initialize_with_markers() after visualization markers
        are set. Creates simulation, viewer, and acquires tensors.
        """
        # Update marker names ordering from visualization markers
        self._marker_names_ordering = (
            list(self._visualization_markers.keys())
            if self._visualization_markers
            else []
        )

        # Create simulation and environments
        self._create_sim(self._visualization_markers)
        self._gym.prepare_sim(self._sim)
        self._enable_viewer_sync = True
        self._viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if not self.headless:
            # subscribe to keyboard shortcuts
            self._viewer = self._gym.create_viewer(self._sim, gymapi.CameraProperties())
            self._gym.subscribe_viewer_keyboard_event(
                self._viewer, gymapi.KEY_Q, "QUIT"
            )
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
            self._gym.subscribe_viewer_keyboard_event(
                self._viewer, gymapi.KEY_M, "toggle_markers"
            )

            # Subscribe to custom key handlers
            self._register_custom_key_handlers()

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
        )[..., 1 : self.scene_lib.num_objects_per_scene + 1, :]
        self._object_indices = torch_utils.to_torch(
            self._object_indices, dtype=torch.int32, device=self.device
        ).view(self.num_envs, self.scene_lib.num_objects_per_scene)

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

        # self._reset_states = RobotState(
        #     dof_pos=torch.zeros(
        #         self.num_envs, self._num_dof, dtype=torch.float, device=self.device
        #     ),
        #     dof_vel=torch.zeros(
        #         self.num_envs, self._num_dof, dtype=torch.float, device=self.device
        #     ),
        #     rigid_body_pos=torch.zeros(
        #         self.num_envs, self._num_bodies, 3, dtype=torch.float, device=self.device
        #     ),
        #     rigid_body_rot=torch.zeros(
        #         self.num_envs, self._num_bodies, 4, dtype=torch.float, device=self.device
        #     ),
        #     rigid_body_vel=torch.zeros(
        #         self.num_envs, self._num_bodies, 3, dtype=torch.float, device=self.device
        #     ),
        #     rigid_body_ang_vel=torch.zeros(
        #         self.num_envs, self._num_bodies, 3, dtype=torch.float, device=self.device
        #     ),
        #     state_conversion=StateConversion.SIMULATOR,
        # )

        self._rigid_body_vel[:, 0, :] = 0
        self._rigid_body_ang_vel[:, 0, :] = 0

        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces = contact_force_tensor.view(
            self.num_envs, bodies_per_env, 3
        )[..., : self._num_bodies, :]

        if self.scene_lib.num_objects_per_scene == 0:
            self._object_contact_forces = None
        else:
            self._object_contact_forces = contact_force_tensor.view(
                self.num_envs, bodies_per_env, 3
            )[..., self._num_bodies :, :]

        if not self.headless:
            self._init_camera()

        if self._visualization_markers:
            self._build_marker_state_tensors()

    def _register_custom_key_handlers(self) -> None:
        """Register custom keyboard event handlers"""
        # Define available keys for custom handlers (1-0 for consistency with IsaacLab)
        available_keys = {
            "1": gymapi.KEY_1,
            "2": gymapi.KEY_2,
            "3": gymapi.KEY_3,
            "4": gymapi.KEY_4,
            "5": gymapi.KEY_5,
            "6": gymapi.KEY_6,
            "7": gymapi.KEY_7,
            "8": gymapi.KEY_8,
            "9": gymapi.KEY_9,
            "0": gymapi.KEY_0,
        }

        # Register custom key handlers
        for key_name, handler in self._custom_key_handlers.items():
            if key_name in available_keys:
                action_name = f"custom_{key_name}"
                self._gym.subscribe_viewer_keyboard_event(
                    self._viewer, available_keys[key_name], action_name
                )
            else:
                print(f"Warning: Key '{key_name}' not available for custom handlers")
                print(f"Available keys: {list(available_keys.keys())}")

    # ===== Group 2: Environment Setup & Configuration =====
    def _create_humanoid_asset_options(self) -> gymapi.AssetOptions:
        """Create asset options for the humanoid with robot config applied.

        Returns:
            gymapi.AssetOptions with default values and robot config overrides applied
        """
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0

        def set_value_if_not_none(prev_value, new_value):
            return new_value if new_value is not None else prev_value

        asset_config_options = [
            "replace_cylinder_with_capsule",
            "thickness",
            "max_angular_velocity",
            "max_linear_velocity",
            "density",
            "angular_damping",
            "linear_damping",
            "disable_gravity",
            "fix_base_link",
        ]
        for option in asset_config_options:
            option_value = set_value_if_not_none(
                getattr(asset_options, option), getattr(self.robot_config.asset, option)
            )
            setattr(asset_options, option, option_value)

        asset_options.collapse_fixed_joints = True  # Always collapse fixed joints. The MJCF file can define whether a fixed joint should not be collapsed.

        return asset_options

    def _load_humanoid_asset(self):
        """Load the humanoid asset from file.

        Returns:
            Loaded asset handle (opaque gymapi handle)
        """
        asset_root = self.robot_config.asset.asset_root
        asset_file = self.robot_config.asset.asset_file_name
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = self._create_humanoid_asset_options()
        return self._gym.load_asset(self._sim, asset_root, asset_file, asset_options)

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

    def _add_terrain(self) -> None:
        print("Adding terrain")
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = 0
        tm_params.transform.p.y = 0
        tm_params.transform.p.z = 0.0
        # IsaacGym only supports "average" friction combine mode (PhysX default)
        assert (
            self.terrain.sim_config.combine_mode == CombineMode.AVERAGE
        ), "IsaacGym only supports average friction combine mode"
        tm_params.static_friction = self.terrain.sim_config.static_friction
        tm_params.dynamic_friction = self.terrain.sim_config.dynamic_friction
        tm_params.restitution = self.terrain.sim_config.restitution

        vertices = self.terrain.vertices
        height_offset = self.terrain.sim_config.height_offset
        vertices[..., 2] += height_offset

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

        gymutil.parse_sim_config(asdict(self.config.sim), sim_params)
        return sim_params

    def _create_sim(
        self,
        visualization_markers: Optional[Dict[str, VisualizationMarkerConfig]] = None,
    ) -> None:
        self._sim_params = self._parse_sim_params()

        # Set Z axis to up.
        self._sim_params.up_axis = gymapi.UP_AXIS_Z
        self._sim_params.gravity.x = 0
        self._sim_params.gravity.y = 0
        self._sim_params.gravity.z = -9.81

        self._physics_engine = gymapi.SIM_PHYSX

        sim = self._gym.create_sim(
            self._graphics_device_id
            if self.device.index is None
            else self.device.index,
            self._graphics_device_id,
            self._physics_engine,
            self._sim_params,
        )
        if sim is None:
            print("*** Failed to create sim")
            quit()

        self._sim = sim

        if self.terrain is not None:
            self._add_terrain()
        self._load_marker_asset()

        self._create_envs(
            0,
            int(np.sqrt(self.num_envs)),
            visualization_markers,
        )

    def _create_envs(
        self,
        spacing: float,
        num_per_row: int,
        visualization_markers: Optional[Dict[str, VisualizationMarkerConfig]] = None,
    ) -> None:
        lower = gymapi.Vec3(0.0, 0.0, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # Load the base humanoid asset
        self._humanoid_asset = humanoid_asset = self._load_humanoid_asset()

        # Create multiple asset variants for friction domain randomization if needed
        self._humanoid_assets_for_friction = self._create_friction_randomized_assets(
            humanoid_asset
        )

        robot_num_bodies = self._gym.get_asset_rigid_body_count(humanoid_asset)
        assert (
            robot_num_bodies == self._num_bodies
        ), f"Number of bodies in the config {self._num_bodies} doesn't match provided robot {robot_num_bodies}"
        self._dof_names = self._gym.get_asset_dof_names(humanoid_asset)
        robot_num_dof = self._gym.get_asset_dof_count(humanoid_asset)
        assert (
            robot_num_dof == len(self._dof_names)
        ), f"Number of dofs in the config {len(self._dof_names)} doesn't match provided robot {robot_num_dof}"
        self._num_joints = self._gym.get_asset_joint_count(humanoid_asset)

        self._humanoid_handles = []
        self._object_handles = []
        self._object_indices = []
        self._envs = []

        self._object_assets = {}
        if self.scene_lib.num_scenes() > 0:
            self._load_object_assets()

        with Progress() as progress:
            task = progress.add_task(
                f"[cyan]Creating {self.num_envs} environments...", total=self.num_envs
            )
            for env_id in range(self.num_envs):
                env_ptr = self._gym.create_env(self._sim, lower, upper, num_per_row)
                # Get the appropriate asset for this environment (for friction domain randomization)
                env_asset = self._get_asset_for_env(env_id)
                self._build_env(env_id, env_ptr, env_asset, visualization_markers)
                self._envs.append(env_ptr)
                progress.update(task, advance=1)

    def _load_object_assets(self) -> None:
        if self.scene_lib.num_scenes() > 0:
            self._object_names = []

            # Count assets
            asset_count = sum(
                1 for scene in self.scene_lib.scenes for obj in scene.objects
            )

            with Progress() as progress:
                task = progress.add_task(
                    "[green]Loading object assets...",
                    total=asset_count,
                )

                # Iterate through all scenes and their objects
                for scene in self.scene_lib.scenes:
                    for obj in scene.objects:
                        # Skip if we've already processed this object type
                        if not obj.is_first_instance:
                            progress.update(task, advance=1)
                            continue

                        first_object_id = obj.first_instance_id
                        if isinstance(obj, PrimitiveSceneObject):
                            # Handle primitive shapes by creating temporary URDFs
                            urdf_path = self._create_primitive_urdf(obj)
                            object_name = os.path.splitext(os.path.basename(urdf_path))[
                                0
                            ]
                            asset_path = urdf_path
                        else:
                            # Handle mesh objects
                            assert isinstance(obj, MeshSceneObject)
                            object_name = os.path.splitext(
                                os.path.basename(obj.object_path)
                            )[0]
                            asset_path = obj.object_path

                        asset_root = os.path.dirname(asset_path)
                        asset_file = os.path.basename(asset_path)

                        # Create asset options
                        object_asset_options = self._create_asset_options(obj)

                        # Load object asset and store in dictionary
                        object_asset = self._gym.load_asset(
                            self._sim, asset_root, asset_file, object_asset_options
                        )

                        # Add force sensor - common for both types
                        sensor_pose = gymapi.Transform()
                        sensor_options = gymapi.ForceSensorProperties()
                        sensor_options.enable_forward_dynamics_forces = False
                        sensor_options.enable_constraint_solver_forces = True
                        sensor_options.use_world_frame = False
                        self._gym.create_asset_force_sensor(
                            object_asset, 0, sensor_pose, sensor_options
                        )
                        self._object_names.append(object_name)
                        self._object_assets[first_object_id] = object_asset

                        progress.update(task, advance=1)

            print(
                f"=========== Total number of unique objects is {len(self._object_assets)}"
            )

    def _create_asset_options(self, obj: SceneObject) -> gymapi.AssetOptions:
        """
        Create asset options for an object based on its configuration.

        Args:
            obj: The SceneObject instance.

        Returns:
            The configured asset options.
        """
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        object_options_dict = obj.options.to_dict()

        if object_options_dict.get("vhacd_enabled", False):
            object_asset_options.vhacd_params = gymapi.VhacdParams()
        for key, value in object_options_dict.items():
            if type(value) is dict:
                if hasattr(object_asset_options, key):
                    object_asset_sub_options = getattr(object_asset_options, key)
                    for sub_key, sub_value in value.items():
                        if hasattr(object_asset_sub_options, sub_key):
                            setattr(
                                object_asset_sub_options,
                                sub_key,
                                sub_value,
                            )
                else:
                    print(f"Warning: {key} is not a valid option for object asset")
            else:
                if hasattr(object_asset_options, key):
                    setattr(object_asset_options, key, value)
                else:
                    print(f"Warning: {key} is not a valid option for object asset")
        return object_asset_options

    def _create_primitive_urdf(self, obj: PrimitiveSceneObject) -> str:
        """
        Create a URDF file for a primitive shape.

        Args:
            obj: The PrimitiveSceneObject instance.

        Returns:
            The path to the created URDF file.
        """
        # Create a temporary directory if it doesn't exist
        temp_dir = self._temp_dir.name

        # Generate a unique filename based on the primitive ID
        urdf_filename = f"{obj.object_identifier}.urdf"
        urdf_path = os.path.join(temp_dir, urdf_filename)

        # Skip if the file already exists
        if os.path.exists(urdf_path):
            return urdf_path

        # Get primitive dimensions from the object
        if isinstance(obj, BoxSceneObject):
            box_size = f"{obj.width} {obj.depth} {obj.height}"
            geometry_xml = f'<box size="{box_size}"/>'
        elif isinstance(obj, SphereSceneObject):
            radius = obj.radius
            geometry_xml = f'<sphere radius="{radius}"/>'
        elif isinstance(obj, CylinderSceneObject):
            radius = obj.radius
            height = obj.height
            geometry_xml = f'<cylinder radius="{radius}" length="{height}"/>'
        else:
            raise ValueError(f"Unsupported primitive shape: {obj.object_identifier}")

        # Create the URDF XML content
        urdf_content = f"""<?xml version="1.0"?>
<robot name="{obj.object_identifier}">
  <link name="base_link">
    <visual>
      <geometry>
        {geometry_xml}
      </geometry>
    </visual>
    <collision>
      <geometry>
        {geometry_xml}
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
</robot>
"""

        # Write the URDF file
        with open(urdf_path, "w") as f:
            f.write(urdf_content)

        return urdf_path

    def _build_env(
        self,
        env_id: int,
        env_ptr,
        humanoid_asset,
        visualization_markers: Optional[Dict[str, VisualizationMarkerConfig]] = None,
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

        for j in range(self._num_bodies):
            self._gym.set_rigid_body_color(
                env_ptr,
                humanoid_handle,
                j,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(0.54, 0.85, 0.2),
            )

        dof_props = self._gym.get_actor_dof_properties(env_ptr, humanoid_handle)

        for dof_name, dof_info in self.robot_config.control.control_info.items():
            if dof_info.effort_limit is not None:
                dof_props["effort"][
                    self.robot_config.kinematic_info.dof_names.index(dof_name)
                ] = dof_info.effort_limit
            if dof_info.velocity_limit is not None:
                dof_props["velocity"][
                    self.robot_config.kinematic_info.dof_names.index(dof_name)
                ] = dof_info.velocity_limit

        if self.control_type == ControlType.BUILT_IN_PD:
            dof_props["driveMode"] = gymapi.DOF_MODE_POS
        else:
            dof_props["driveMode"] = gymapi.DOF_MODE_EFFORT

        # Set PD gains for built-in PD controller
        for i in range(self._num_dof):
            dof_name = self.robot_config.kinematic_info.dof_names[i]
            stiffness = self.robot_config.control.control_info[dof_name].stiffness
            damping = self.robot_config.control.control_info[dof_name].damping
            armature = self.robot_config.control.control_info[dof_name].armature
            friction = self.robot_config.control.control_info[dof_name].friction
            if not self.control_type == ControlType.BUILT_IN_PD:
                # Manual PD controller handles stiffness-damping.
                # This disables additional stiffness-damping that is added by the built-in PD controller.
                stiffness = 0.0
                damping = 0.0

            dof_props["stiffness"][i] = stiffness
            dof_props["damping"][i] = damping
            dof_props["armature"][i] = armature
            dof_props["friction"][i] = friction

        self._gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_props)

        # dof_props_debug = self._gym.get_actor_dof_properties(env_ptr, humanoid_handle)
        # # ('hasLimits', 'lower', 'upper', 'driveMode', 'velocity', 'effort', 'stiffness', 'damping', 'friction', 'armature')
        # # driveMode: Drive mode (0=None, 1=Position, 2=Velocity, 3=Effort)
        # print(f"=========== dof_props field names: {dof_props_debug.dtype.names}")
        # print(f"=========== dof_props: \n {dof_props_debug}")

        # Apply COM domain randomization to this specific actor (must be done right after actor creation)
        self._apply_com_domain_randomization_to_actor(env_ptr, humanoid_handle, env_id)

        self._humanoid_handles.append(humanoid_handle)

        if self.scene_lib.num_objects_per_scene > 0:
            self._build_object_playground(env_id, env_ptr)

        self._build_markers(env_id, env_ptr, visualization_markers)

    def _build_object_playground(self, env_id: int, env_ptr) -> None:
        """
        Build playground of scene objects. Different scenes can be created for different envs.

        Args:
            env_id: Environment ID to build objects for.
            env_ptr: Environment pointer.
        """

        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        scene = self.scene_lib.scenes[env_id]

        # Get scene offset for dummy spawn position
        scene_offset_x, scene_offset_y = self.scene_lib.scene_offsets[env_id]

        # Process each object in the scene
        # Objects spawned at scene offset (x,y) with z=0 - actual positions set via reset_envs()
        for obj_idx, obj in enumerate(scene.objects):
            # Get the asset directly using first_instance_id
            object_asset = self._object_assets[obj.first_instance_id]

            # Spawn at scene offset to avoid collision, actual pose set via reset
            object_pose = gymapi.Transform()
            object_pose.p = gymapi.Vec3(scene_offset_x, scene_offset_y, 0.0)
            object_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            # Create object instance in the environment
            object_name = f"object_{obj_idx}"
            object_handle = self._gym.create_actor(
                env_ptr,
                object_asset,
                object_pose,
                object_name,
                col_group,
                col_filter,
                segmentation_id,
            )

            # Store handle and index for this object
            self._object_handles.append(object_handle)
            object_idx = self._gym.get_actor_index(
                env_ptr, object_handle, gymapi.DOMAIN_SIM
            )
            self._object_indices.append(object_idx)

            # Apply texture if specified in object options
            if hasattr(obj, "options") and obj.options is not None:
                texture_path = obj.options.to_dict().get("texture_path")
                if texture_path and not self.headless:
                    self._apply_texture_to_object(env_ptr, object_handle, texture_path)

    def _apply_texture_to_object(
        self, env_ptr, object_handle, texture_path: str
    ) -> None:
        """
        Apply a texture to an object's rigid bodies.

        Args:
            env_ptr: Environment pointer
            object_handle: Handle to the object/actor
            texture_path: Path to the texture file (absolute or relative)
        """
        # Load texture if not already in cache
        if texture_path not in self._texture_handles:
            # Expand path if it's relative
            if not os.path.isabs(texture_path):
                # Try to resolve relative to project root or as-is
                abs_path = os.path.abspath(texture_path)
                if os.path.exists(abs_path):
                    texture_path = abs_path

            if os.path.exists(texture_path):
                texture_handle = self._gym.create_texture_from_file(
                    self._sim, texture_path
                )

                if texture_handle == gymapi.INVALID_HANDLE:
                    print(f"Warning: Failed to load texture from {texture_path}")
                    return
                else:
                    self._texture_handles[texture_path] = texture_handle
                    print(f"Loaded texture: {os.path.basename(texture_path)}")
            else:
                print(f"Warning: Texture file not found: {texture_path}")
                return

        # Apply texture to all rigid bodies of this object
        texture_handle = self._texture_handles.get(texture_path)
        if texture_handle is not None:
            num_bodies = self._gym.get_actor_rigid_body_count(env_ptr, object_handle)
            for body_idx in range(num_bodies):
                self._gym.set_rigid_body_texture(
                    env_ptr,
                    object_handle,
                    body_idx,
                    gymapi.MESH_VISUAL,  # Apply to visual mesh only
                    texture_handle,
                )

    def _build_markers(
        self,
        env_id: int,
        env_ptr,
        visualization_markers: Optional[Dict[str, VisualizationMarkerConfig]] = None,
    ) -> None:
        """Build visualization markers for the environment.

        Args:
            env_id (int): Environment ID to build markers for
            env_ptr: Environment pointer from IsaacGym
            visualization_markers (Dict[str, VisualizationMarkerConfig]): Dictionary mapping marker names to their configurations
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
        if self.scene_lib.num_objects_per_scene > 0:
            self._marker_states = self._root_states.view(
                self.num_envs, num_actors, self._root_states.shape[-1]
            )[..., 1 + self.scene_lib.num_objects_per_scene :, :]
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
        # For BUILT_IN_PD, set targets once before loop (efficiency)
        # For PROPORTIONAL/TORQUE, apply inside loop (needs fresh DOF state each substep)
        if self.control_type == ControlType.BUILT_IN_PD:
            self._apply_control()
        for i in range(self.decimation):
            if self.control_type != ControlType.BUILT_IN_PD:
                self._apply_control()
            self._simulate()
            if self.device.type == "cpu":
                self._gym.fetch_results(self._sim, True)
            if self.control_type != ControlType.BUILT_IN_PD:
                self._gym.refresh_dof_state_tensor(self._sim)
        self._refresh_sim_tensors()

    def _refresh_sim_tensors(self) -> None:
        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)
        self._gym.refresh_force_sensor_tensor(self._sim)
        self._gym.refresh_dof_force_tensor(self._sim)
        self._gym.refresh_net_contact_force_tensor(self._sim)

    def _set_simulator_env_state(
        self,
        new_states: ResetState,
        new_object_states: ObjectState = None,
        env_ids: torch.Tensor = None,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Set new states
        self._humanoid_root_states[env_ids, 0:3] = new_states.root_pos
        self._humanoid_root_states[env_ids, 3:7] = new_states.root_rot
        self._humanoid_root_states[env_ids, 7:10] = new_states.root_vel
        self._humanoid_root_states[env_ids, 10:13] = new_states.root_ang_vel

        self._dof_pos[env_ids] = new_states.dof_pos
        self._dof_vel[env_ids] = new_states.dof_vel

        actor_ids = self._humanoid_actor_ids[env_ids]
        set_root_state_ids = actor_ids

        if new_object_states is not None:
            self._object_root_states[env_ids, :, 0:3] = new_object_states.root_pos
            self._object_root_states[env_ids, :, 3:7] = new_object_states.root_rot
            self._object_root_states[env_ids, :, 7:10] = new_object_states.root_vel
            self._object_root_states[env_ids, :, 10:13] = new_object_states.root_ang_vel

            object_ids = self._object_indices[env_ids].flatten()
            set_root_state_ids = torch.cat((set_root_state_ids, object_ids), dim=0)

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

        if self.control_type == ControlType.BUILT_IN_PD:
            self._gym.set_dof_position_target_tensor_indexed(
                self._sim,
                gymtorch.unwrap_tensor(self._dof_pos.contiguous()),
                gymtorch.unwrap_tensor(actor_ids),
                len(actor_ids),
            )

        self._refresh_sim_tensors()

    def _simulate(self) -> None:
        self._gym.simulate(self._sim)
        if self.device == "cpu":
            self._gym.fetch_results(self._sim, True)
        self._gym.refresh_dof_state_tensor(self._sim)

    # ===== Group 4: State Getters =====
    def _get_num_actors_per_env(self) -> int:
        num_actors = self._root_states.shape[0] // self.num_envs
        return num_actors

    def _get_sim_body_ordering(self) -> SimBodyOrdering:
        return SimBodyOrdering(
            body_names=self._gym.get_asset_rigid_body_names(self._humanoid_asset),
            dof_names=self._gym.get_asset_dof_names(self._humanoid_asset),
        )

    def _get_simulator_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RootOnlyState:
        root_pos = self._humanoid_root_states[..., :3].clone()
        root_rot = self._humanoid_root_states[..., 3:7].clone()
        root_vel = self._humanoid_root_states[..., 7:10].clone()
        root_ang_vel = self._humanoid_root_states[..., 10:13].clone()
        if env_ids is not None:
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            root_vel = root_vel[env_ids]
            root_ang_vel = root_ang_vel[env_ids]
        return RootOnlyState(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_object_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> ObjectState:
        root_pos = self._object_root_states[..., :3].clone()
        root_rot = self._object_root_states[..., 3:7].clone()
        root_vel = self._object_root_states[..., 7:10].clone()
        root_ang_vel = self._object_root_states[..., 10:13].clone()
        if env_ids is not None:
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            root_vel = root_vel[env_ids]
            root_ang_vel = root_ang_vel[env_ids]
        return ObjectState(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_bodies_contact_buf(self, env_ids=None) -> RobotState:
        contact_forces = self._contact_forces.clone()
        if env_ids is not None:
            contact_forces = contact_forces[env_ids]
        return RobotState(
            rigid_body_contact_forces=contact_forces,
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_object_contact_buf(self, env_ids=None) -> ObjectState:
        object_contact_forces = self._object_contact_forces.clone()
        if env_ids is not None:
            object_contact_forces = object_contact_forces[env_ids]
        return ObjectState(
            contact_forces=object_contact_forces,
            state_conversion=StateConversion.SIMULATOR,
        )

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
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_dof_forces(self, env_ids=None) -> RobotState:
        dof_forces = self._dof_force_tensor.clone()
        if env_ids is not None:
            dof_forces = dof_forces[env_ids]
        return RobotState(
            dof_forces=dof_forces, state_conversion=StateConversion.SIMULATOR
        )

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
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_dof_limits_for_verification(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve DOF limits from IsaacGym's internal API for verification purposes only.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of (lower_limits, upper_limits)
                                              in IsaacGym's DOF ordering.
        """
        # Extract limits from the first environment
        dof_prop = self._gym.get_actor_dof_properties(
            self._envs[0], self._humanoid_handles[0]
        )
        dof_limits_lower = []
        dof_limits_upper = []
        for j in range(len(dof_prop["upper"])):
            if dof_prop["lower"][j] > dof_prop["upper"][j]:
                dof_limits_lower.append(dof_prop["upper"][j])
                dof_limits_upper.append(dof_prop["lower"][j])
            else:
                dof_limits_lower.append(dof_prop["lower"][j])
                dof_limits_upper.append(dof_prop["upper"][j])

        return (
            torch_utils.to_torch(dof_limits_lower, device=self.device),
            torch_utils.to_torch(dof_limits_upper, device=self.device),
        )

    # ===== Group 5: Control & Computation Methods =====
    def _apply_simulator_pd_targets(self, pd_targets: torch.Tensor) -> None:
        """Applies PD position targets using IsaacGym's internal PD controller."""
        pd_tar_tensor = gymtorch.unwrap_tensor(pd_targets)
        self._gym.set_dof_position_target_tensor(self._sim, pd_tar_tensor)

    def _apply_simulator_torques(self, torques: torch.Tensor) -> None:
        """Applies torques to the robot DOFs."""
        torques_tensor = gymtorch.unwrap_tensor(torques)
        self._gym.set_dof_actuation_force_tensor(self._sim, torques_tensor)

    def _apply_root_velocity_impulse(
        self,
        linear_velocity: torch.Tensor,
        angular_velocity: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Apply velocity impulse to robot root by adding to current velocities."""
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._humanoid_root_states[env_ids, 7:10] += linear_velocity
        self._humanoid_root_states[env_ids, 10:13] += angular_velocity
        
        actor_ids = self._humanoid_actor_ids[env_ids]
        self._gym.set_actor_root_state_tensor_indexed(
            self._sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids),
        )

    # ===== Group 6: Domain Randomization =====
    # - IsaacGym: Must set friction on asset before actor creation
    #   Solution: Create min(num_buckets, num_envs) assets, evenly distribute to environments

    def _create_friction_randomized_assets(self, base_asset) -> List:
        """Create multiple asset copies with different friction/restitution values for domain randomization.

        Due to IsaacGym limitation, friction must be set on assets before actor creation.
        We create min(num_buckets, num_envs) asset variants and distribute them across environments.
        Matches IsaacLab's approach of creating min(num_buckets, num_envs) samples.

        Both IsaacGym and IsaacLab use "average" friction combine mode (PhysX default):
            effective_friction = (robot_friction + terrain_friction) / 2

        Note: IsaacGym only has single friction property (not separate static/dynamic like IsaacLab).
        We use static_friction as the representative value.

        # TODO: in IsaacGym, we apply friction to ALL shapes, not per-body, body_indices is not used.

        Returns:
            List of asset handles, one per friction sample. Returns [base_asset] if no friction DR.
        """
        if (
            self._domain_randomization is None
            or "friction" not in self._domain_randomization
        ):
            return [base_asset]  # No friction randomization, use single asset

        # Note: base simulator already creates min(num_buckets, num_envs) samples
        num_assets_to_create = self._domain_randomization["friction"][
            "static_friction"
        ].shape[0]
        # body_indices stored for reference but not used (we apply friction to all shapes)
        # body_indices = self._domain_randomization["friction"]["body_indices"]

        print(
            f"Creating {num_assets_to_create} asset variants for friction domain randomization"
        )

        assets = []
        for i in range(num_assets_to_create):
            # Load a fresh asset for each friction variant
            asset = self._load_humanoid_asset()

            # Apply friction and restitution for this asset (using index i directly, since samples are pre-randomized)
            # Note: base simulator already sampled random values in _process_friction_domain_randomization
            shape_props = self._gym.get_asset_rigid_shape_properties(asset)

            # Note: In IsaacGym, we apply friction to ALL shapes, not per-body
            # The body_indices tell us which bodies we want to randomize, but we need to
            # apply the sampled friction value to all shapes uniformly for simplicity
            # (Mapping body->shape indices is complex and not exposed in IsaacGym API)

            # Use the first body's randomized values for all shapes (simplified approach)
            # For most configs like body_names=[".*"], all bodies get the same randomization anyway
            sampled_friction = self._domain_randomization["friction"][
                "static_friction"
            ][i, 0].item()
            sampled_restitution = self._domain_randomization["friction"]["restitution"][
                i, 0
            ].item()

            for shape_prop in shape_props:
                # Use pre-randomized friction value directly (no adjustment needed - both sims use average mode)
                # Note: IsaacGym only has single friction property, not separate static/dynamic
                shape_prop.friction = sampled_friction
                shape_prop.restitution = sampled_restitution

            self._gym.set_asset_rigid_shape_properties(asset, shape_props)
            assets.append(asset)

        print(f"Created {len(assets)} friction-randomized asset variants")
        return assets

    def _get_asset_for_env(self, env_id: int):
        """Get the appropriate asset for a given environment ID.

        For friction domain randomization, evenly distributes asset buckets across environments.
        This ensures each friction bucket is used approximately equally.

        Args:
            env_id: Environment ID

        Returns:
            Asset handle to use for this environment
        """
        if len(self._humanoid_assets_for_friction) == 1:
            # No friction randomization, single asset
            return self._humanoid_assets_for_friction[0]

        # Evenly distribute buckets across environments using modulo
        # e.g., 64 buckets for 4096 envs: env 0->bucket 0, env 1->bucket 1, ..., env 64->bucket 0, etc.
        num_buckets = len(self._humanoid_assets_for_friction)
        bucket_id = env_id % num_buckets
        return self._humanoid_assets_for_friction[bucket_id]

    def _update_body_com_and_inertia(self, body_prop, offset: List[float]) -> None:
        """
        Update a single rigid body's Center of Mass and inertia tensor using parallel axis theorem.

        Args:
            body_prop: Single rigid body property object from IsaacGym
            offset: [dx, dy, dz] offset in meters for the Center of Mass

        The parallel axis theorem: I_new = I_old + m * (r²·I - r⊗r)
        Where:
        - I_new: new inertia tensor about new axis
        - I_old: original inertia tensor about center of mass
        - m: mass of the body
        - r: offset vector [dx, dy, dz]
        - r²: dot product r·r
        - r⊗r: outer product of r
        """

        mass = body_prop.mass
        dx, dy, dz = offset

        # Update Center of Mass
        body_prop.com.x += dx
        body_prop.com.y += dy
        body_prop.com.z += dz

        # Apply parallel axis theorem to inertia tensor
        # Original inertia matrix (IsaacGym stores as [Ixx, Iyy, Izz, Ixy, Ixz, Iyz])
        Ixx_old = body_prop.inertia.x.x
        Iyy_old = body_prop.inertia.y.y
        Izz_old = body_prop.inertia.z.z
        Ixy_old = body_prop.inertia.x.y
        Ixz_old = body_prop.inertia.x.z
        Iyz_old = body_prop.inertia.y.z

        # Parallel axis theorem corrections
        r_squared = dx * dx + dy * dy + dz * dz

        # New inertia components
        Ixx_new = Ixx_old + mass * (
            r_squared - dx * dx
        )  # = Ixx_old + mass * (dy² + dz²)
        Iyy_new = Iyy_old + mass * (
            r_squared - dy * dy
        )  # = Iyy_old + mass * (dx² + dz²)
        Izz_new = Izz_old + mass * (
            r_squared - dz * dz
        )  # = Izz_old + mass * (dx² + dy²)
        Ixy_new = Ixy_old - mass * dx * dy
        Ixz_new = Ixz_old - mass * dx * dz
        Iyz_new = Iyz_old - mass * dy * dz

        # Update inertia tensor
        body_prop.inertia.x.x = Ixx_new
        body_prop.inertia.y.y = Iyy_new
        body_prop.inertia.z.z = Izz_new
        body_prop.inertia.x.y = Ixy_new
        body_prop.inertia.y.x = Ixy_new  # Symmetric
        body_prop.inertia.x.z = Ixz_new
        body_prop.inertia.z.x = Ixz_new  # Symmetric
        body_prop.inertia.y.z = Iyz_new
        body_prop.inertia.z.y = Iyz_new  # Symmetric

    def _apply_com_domain_randomization_to_actor(
        self, env_ptr, humanoid_handle, env_id: int
    ) -> None:
        """Apply center of mass domain randomization to a specific actor.

        IMPORTANT: In IsaacGym, this must be called right after actor creation in _build_env().

        Args:
            env_ptr: Environment pointer
            humanoid_handle: Actor handle for the humanoid
            env_id: Environment ID for this actor
        """
        if (
            self._domain_randomization is None
            or "center_of_mass" not in self._domain_randomization
        ):
            return

        body_props = self._gym.get_actor_rigid_body_properties(env_ptr, humanoid_handle)

        body_indices = self._domain_randomization["center_of_mass"]["body_indices"]
        com_offsets = self._domain_randomization["center_of_mass"]["com"][env_id]

        for idx, body_idx in enumerate(body_indices):
            offset = com_offsets[idx].cpu().numpy().tolist()
            self._update_body_com_and_inertia(body_props[body_idx], offset)

        self._gym.set_actor_rigid_body_properties(
            env_ptr, humanoid_handle, body_props, recomputeInertia=False
        )

    # ===== Group 7: Rendering & Visualization =====
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
                elif evt.action == "toggle_markers" and evt.value > 0:
                    self._toggle_markers()
                elif evt.action.startswith("custom_") and evt.value > 0:
                    # Handle custom key events
                    key_name = evt.action[7:]  # Remove "custom_" prefix
                    if key_name in self._custom_key_handlers:
                        try:
                            self._custom_key_handlers[key_name]()
                        except Exception as e:
                            print(
                                f"Error executing custom key handler for '{key_name}': {e}"
                            )

            if self.device.type != "cpu":
                self._gym.fetch_results(self._sim, True)

            if self._enable_viewer_sync:
                self._gym.step_graphics(self._sim)
                self._gym.draw_viewer(self._viewer, self._sim, True)
            else:
                self._gym.poll_viewer_events(self._viewer)
        super().render()

    def _update_simulator_markers(
        self, markers_state: Optional[Dict[str, MarkerState]] = None
    ) -> None:
        if markers_state is None or len(self._marker_names_ordering) == 0:
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
        super().close()
        if self._viewer:
            self._gym.destroy_viewer(self._viewer)
        self._gym.destroy_sim(self._sim)

        # Clean up the temporary directory
        print("Cleaning up the temporary directory")
        print(self._temp_dir)
        self._temp_dir.cleanup()

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

        current_cam_transform = self._gym.get_viewer_camera_transform(
            self._viewer, None
        )
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
