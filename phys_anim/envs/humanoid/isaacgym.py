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
import os.path as osp

import numpy as np
from isaacgym import gymapi, gymtorch, gymutil  # type: ignore[misc]
import torch
from torch import Tensor
from hydra.utils import instantiate
from easydict import EasyDict
from rich.progress import Progress

from isaac_utils import rotations, torch_utils
from phys_anim.envs.humanoid.common import BaseHumanoid
from phys_anim.envs.base_interface.utils import build_pd_action_offset_scale
from phys_anim.envs.base_interface.isaacgym import GymBaseInterface
from phys_anim.utils.file_utils import load_yaml
from phys_anim.utils.motion_lib import MotionLib


class Humanoid(BaseHumanoid, GymBaseInterface):  # type: ignore[misc]
    def __init__(self, config, device: torch.device):
        self.w_last = True  # quaternion structure in isaacgym
        self.config = config
        self.device = device
        self.sim_params = self.parse_sim_params()
        self.physics_engine = gymapi.SIM_PHYSX

        self.plane_static_friction = self.config.simulator.plane.static_friction
        self.plane_dynamic_friction = self.config.simulator.plane.dynamic_friction
        self.plane_restitution = self.config.simulator.plane.restitution

        super().__init__(config, device)
        assert (
            self.dof_offsets[-1] == self.num_dof
        ), f"Mismatch in num DOFs {self.num_dof} and {self.dof_offsets[-1]}"

        self.dt: float = self.config.simulator.sim.control_freq_inv * self.sim_params.dt

        # Refresh tensors BEFORE we acquire them https://forums.developer.nvidia.com/t/isaacgym-preview-4-actor-root-state-returns-nans-with-isaacgymenvs-style-task/223738/4
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        sensors_per_env = 2
        self.vec_sensor_tensor: Tensor = gymtorch.wrap_tensor(sensor_tensor).view(
            self.num_envs, sensors_per_env * 6
        )

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor: Tensor = gymtorch.wrap_tensor(dof_force_tensor).view(
            self.num_envs, self.num_dof
        )

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.root_states: Tensor = gymtorch.wrap_tensor(actor_root_state)

        self.object_root_states = self.root_states[-self.total_num_objects :]
        self.object_indices = torch_utils.to_torch(
            self.object_indices, dtype=torch.int32, device=self.device
        )

        num_actors = self.get_num_actors_per_env()

        if self.total_num_objects == 0:
            self.humanoid_root_states = self.root_states.view(
                self.num_envs, num_actors, actor_root_state.shape[-1]
            )[..., 0, :]
        else:
            self.humanoid_root_states = self.root_states[
                : -self.total_num_objects
            ].view(self.num_envs, num_actors, actor_root_state.shape[-1])[..., 0, :]

        self.initial_humanoid_root_states = self.humanoid_root_states.clone()
        self.initial_humanoid_root_states[:, 7:13] = 0

        self.humanoid_actor_ids = num_actors * torch.arange(
            self.num_envs, device=self.device, dtype=torch.int32
        )

        # create some wrapper tensors for different slices
        self.dof_state: Tensor = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self.dof_state.shape[0] // self.num_envs
        self.dof_pos = self.dof_state.view(self.num_envs, dofs_per_env, 2)[
            ..., : self.num_dof, 0
        ]
        self.dof_vel = self.dof_state.view(self.num_envs, dofs_per_env, 2)[
            ..., : self.num_dof, 1
        ]

        self.initial_dof_pos = self.dof_pos.clone()
        self.initial_dof_vel = self.dof_vel.clone()

        if self.total_num_objects == 0:
            self.rigid_body_state: Tensor = gymtorch.wrap_tensor(rigid_body_state)
        else:
            self.rigid_body_state: Tensor = gymtorch.wrap_tensor(rigid_body_state)[
                : -self.total_num_objects
            ]

        bodies_per_env = self.rigid_body_state.shape[0] // self.num_envs
        rigid_body_state_reshaped = self.rigid_body_state.view(
            self.num_envs, bodies_per_env, 13
        )

        self.rigid_body_pos = rigid_body_state_reshaped[..., : self.num_bodies, 0:3]
        self.rigid_body_rot = rigid_body_state_reshaped[..., : self.num_bodies, 3:7]
        self.rigid_body_vel = rigid_body_state_reshaped[..., : self.num_bodies, 7:10]
        self.rigid_body_ang_vel = rigid_body_state_reshaped[
            ..., : self.num_bodies, 10:13
        ]

        self.initial_rigid_body_pos = self.rigid_body_pos.clone()
        self.initial_rigid_body_rot = self.rigid_body_rot.clone()
        self.initial_rigid_body_vel = self.rigid_body_vel.clone()
        self.initial_rigid_body_ang_vel = self.rigid_body_ang_vel.clone()

        if self.total_num_objects == 0:
            contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        else:
            contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)[
                : -self.total_num_objects
            ]
        self.contact_forces = contact_force_tensor.view(
            self.num_envs, bodies_per_env, 3
        )[..., : self.num_bodies, :]

        self.key_body_ids = self.build_body_ids_tensor(self.config.robot.key_bodies)
        self.contact_body_ids = self.build_body_ids_tensor(
            self.config.robot.contact_bodies
        )

        props = self.gym.get_asset_dof_properties(self.humanoid_asset)
        self.process_dof_props(props)
        self.create_legged_robot_tensors()

        if self.viewer is not None:
            self.init_camera()

        self.export_video: bool = self.config.export_video

        if self.export_video:
            self.setup_cameras()

        self.export_motion: bool = self.config.export_motion

        if self.export_motion:
            self.motion_recording = {}

        # Allows the agent to disable resets temporarily.
        self.disable_reset = False

        self.build_termination_heights()

        # Call at the end to enable base_interface classes to generate the required base_interface tensors.
        self.on_environment_ready()

    ###############################################################
    # Set up IsaacGym environment
    ###############################################################
    def parse_sim_params(self):
        # initialize sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / self.config.simulator.sim.fps
        sim_params.num_client_threads = 0

        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = True
        sim_params.physx.num_subscenes = 0
        sim_params.physx.max_gpu_contact_pairs = (
            self.config.robot.contact_pairs_multiplier * 1024 * 1024
        )
        sim_params.use_gpu_pipeline = True

        gymutil.parse_sim_config(self.config.simulator.sim, sim_params)
        return sim_params

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, "z")
        super().create_sim()

        self.create_ground_plane()
        self.create_envs(
            # Force zero spacing. Our terrain and scene_lib class handle spawning and object-humanoid allocations.
            self.num_envs,
            0,
            int(np.sqrt(self.num_envs)),
        )

    def create_ground_plane(self):
        print("Creating ground plane")
        if self.config.terrain is None:
            self.add_default_ground()
        else:
            self.add_terrain()
        print("Ground plane created")

    def add_default_ground(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

    def add_terrain(self):
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = 0
        tm_params.transform.p.y = 0
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.config.simulator.plane.static_friction
        tm_params.dynamic_friction = self.config.simulator.plane.dynamic_friction
        tm_params.restitution = self.config.simulator.plane.restitution
        self.gym.add_triangle_mesh(
            self.sim,
            self.terrain.vertices.flatten(order="C"),
            self.terrain.triangles.flatten(order="C"),
            tm_params,
        )

    def create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(0.0, 0.0, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = self.config.robot.asset.asset_root
        asset_file = self.config.robot.asset.asset_file_name

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
                getattr(asset_options, option), getattr(self.config.robot.asset, option)
            )
            setattr(asset_options, option, option_value)

        self.humanoid_asset = humanoid_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )

        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        assert (
            self.num_bodies == self.config.robot.num_bodies
        ), f"Number of bodies in the config {self.config.robot.num_bodies} doesn't match provided robot {self.num_bodies}"
        self.body_names = self.gym.get_asset_rigid_body_names(humanoid_asset)
        self.dof_names = self.gym.get_asset_dof_names(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        if "h1" in asset_file:
            motor_efforts = [360] * self.num_act
        else:
            actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
            motor_efforts = [prop.motor_effort for prop in actuator_props]

        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(
            humanoid_asset, self.config.robot.right_foot_name
        )
        left_foot_idx = self.gym.find_asset_rigid_body_index(
            humanoid_asset, self.config.robot.left_foot_name
        )
        sensor_pose = gymapi.Transform()

        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = torch_utils.to_torch(motor_efforts, device=self.device)

        self.humanoid_handles = []
        self.object_handles = []
        self.object_indices = []
        self.envs = []
        self.object_envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        self.object_assets = []
        if self.config.scene_lib is not None:
            self.load_object_assets()

        with Progress() as progress:
            task = progress.add_task(
                f"[cyan]Creating {self.num_envs} environments...", total=self.num_envs
            )
            for i in range(self.num_envs):
                # create env instance
                env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
                self.build_env(i, env_ptr, humanoid_asset)
                self.envs.append(env_ptr)

                progress.update(task, advance=1)

        if len(self.object_assets) > 0:
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.build_object_playground(env_ptr)
            self.object_envs.append(env_ptr)

        dof_prop = self.gym.get_actor_dof_properties(
            self.envs[0], self.humanoid_handles[0]
        )
        for j in range(self.num_dof):
            if dof_prop["lower"][j] > dof_prop["upper"][j]:
                self.dof_limits_lower.append(dof_prop["upper"][j])
                self.dof_limits_upper.append(dof_prop["lower"][j])
            else:
                self.dof_limits_lower.append(dof_prop["lower"][j])
                self.dof_limits_upper.append(dof_prop["upper"][j])

        self.dof_limits_lower = torch_utils.to_torch(
            self.dof_limits_lower, device=self.device
        )
        self.dof_limits_upper = torch_utils.to_torch(
            self.dof_limits_upper, device=self.device
        )

        if self.isaac_pd:
            self._pd_action_offset, self._pd_action_scale = (
                build_pd_action_offset_scale(
                    self.dof_offsets,
                    self.dof_limits_lower,
                    self.dof_limits_upper,
                    self.device,
                    self.gym.get_asset_dof_names(humanoid_asset),
                )
            )

    def load_object_assets(self):
        if self.scene_lib.total_spawned_scenes > 0:
            self.object_target_positions = []
            self.object_names = []

            with Progress() as progress:
                task = progress.add_task(
                    "[green]Loading object assets...",
                    total=len(self.scene_lib.object_spawn_list),
                )

                for i, object_info in enumerate(self.scene_lib.object_spawn_list):
                    object_name = os.path.splitext(
                        os.path.basename(object_info.object_path)
                    )[0]

                    object_asset_options = gymapi.AssetOptions()
                    for key, value in object_info.object_options.items():
                        if isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                setattr(
                                    getattr(object_asset_options, key),
                                    sub_key,
                                    sub_value,
                                )
                        else:
                            setattr(object_asset_options, key, value)

                    # Load Asset
                    object_asset = self.gym.load_asset(
                        self.sim,
                        os.path.dirname(object_info.object_path),
                        f"{object_name}.urdf",
                        object_asset_options,
                    )
                    self.object_assets.append(object_asset)

                    # Load Joint Target Positions
                    yaml_path = os.path.join(
                        os.path.dirname(object_info.object_path), f"{object_name}.yaml"
                    )
                    if os.path.exists(yaml_path):
                        target_position = load_yaml(yaml_path).get("hip", [0, 0, 0])
                    else:
                        target_position = [
                            0,
                            0,
                            0,
                        ]  # Default position if YAML doesn't exist
                    self.object_target_positions.append(
                        torch.tensor(
                            target_position, device=self.device, dtype=torch.float
                        ).view(-1)
                    )

                    self.object_names.append(object_name)

                    progress.update(task, advance=1)

            print(
                f"=========== Total number of unique objects is {len(self.object_assets)}"
            )

    def build_env(self, env_id, env_ptr, humanoid_asset):
        col_group = env_id
        col_filter = 0 if self.config.robot.asset.self_collisions else 1
        segmentation_id = 0

        start_pose = gymapi.Transform()
        asset_file = self.config.robot.asset.asset_file_name
        if (
            asset_file == "mjcf/ov_humanoid.xml"
            or asset_file == "mjcf/ov_humanoid_sword_shield.xml"
        ):
            char_h = 0.927
        else:
            char_h = 0.89

        # Space out the humanoids on initial spawn.
        start_offset = [env_id, env_id, env_id]
        start_offset[self.up_axis_idx] = char_h
        start_pose.p = gymapi.Vec3(*start_offset)

        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        humanoid_handle = self.gym.create_actor(
            env_ptr,
            humanoid_asset,
            start_pose,
            "humanoid",
            col_group,
            col_filter,
            segmentation_id,
        )

        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

        humanoid_mass = np.sum(
            [
                prop.mass
                for prop in self.gym.get_actor_rigid_body_properties(
                    env_ptr, humanoid_handle
                )
            ]
        )

        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(
                env_ptr,
                humanoid_handle,
                j,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(0.54, 0.85, 0.2),
            )

        dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
        if self.isaac_pd:
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            if self.config.robot.control.isaac_pd_scale:
                pd_scale = humanoid_mass / self.config.robot.default_humanoid_mass
                dof_prop["stiffness"] *= pd_scale
                dof_prop["damping"] *= pd_scale
        else:
            dof_prop["driveMode"] = gymapi.DOF_MODE_EFFORT

        self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)

        filter_ints = self.config.robot.asset.filter_ints
        if filter_ints is not None:
            props = self.gym.get_actor_rigid_shape_properties(env_ptr, humanoid_handle)

            assert len(filter_ints) == len(props)
            for p_idx in range(len(props)):
                props[p_idx].filter = filter_ints[p_idx]

            self.gym.set_actor_rigid_shape_properties(env_ptr, humanoid_handle, props)

        self.humanoid_handles.append(humanoid_handle)

    def build_object_playground(self, env_ptr):
        print("=========== Building object playground")
        import trimesh

        from phys_anim.envs.env_utils.object_utils import (
            as_mesh,
            compute_bounding_box,
            get_object_heightmap,
        )

        total_objects = sum(len(scene["objects"]) for scene in self.scene_lib.scenes)
        with Progress() as progress:
            task = progress.add_task("[cyan]Spawning objects...", total=total_objects)
            for scene_idx, scene_spawn_info in enumerate(self.scene_lib.scenes):
                scene_offset = self.scene_lib.scene_offsets[scene_idx]

                height_at_scene_origin = self.get_ground_heights(
                    torch.tensor(
                        [[scene_offset[0], scene_offset[1]]],
                        device=self.device,
                        dtype=torch.float,
                    )
                ).item()
                self.scene_position.append(
                    torch.tensor(
                        [scene_offset[0], scene_offset[1], height_at_scene_origin],
                        device=self.device,
                        dtype=torch.float,
                    )
                )

                for obj in scene_spawn_info["objects"]:
                    progress.update(
                        task,
                        advance=1,
                        description=f"[cyan]Spawning {obj['path'].split('/')[-1]}",
                    )
                    object_id = obj["id"]
                    object_spawn_info = self.scene_lib.object_spawn_list[object_id]

                    self.total_num_objects += 1

                    object_asset = self.object_assets[
                        self.scene_lib.object_path_to_id[object_spawn_info.object_path]
                    ]
                    object_name = object_spawn_info.object_path.split("/")[-1].split(
                        "."
                    )[0]
                    object_pose = gymapi.Transform()

                    initial_object_pose = self.scene_lib.get_object_pose(
                        torch.tensor([object_id], device=self.device, dtype=torch.int),
                        torch.tensor([0.0], device=self.device, dtype=torch.float),
                    )

                    # Calculate the global position of the object
                    global_object_position = torch.tensor(
                        [
                            scene_offset[0] + initial_object_pose.translations[0, 0],
                            scene_offset[1] + initial_object_pose.translations[0, 1],
                            0,  # We'll set the z-coordinate later
                        ],
                        device=self.device,
                        dtype=torch.float,
                    )

                    # Convert global position to terrain map coordinates
                    terrain_coords = (
                        global_object_position[:2] / self.terrain.horizontal_scale
                    ).long()

                    # Assert that the object is within the valid range of the height samples
                    assert (
                        0 <= terrain_coords[0] < self.height_samples.shape[0] - 2
                    ), f"Scene {scene_idx}: Object {object_name} is outside the valid range of height samples (x-axis)"
                    assert (
                        0 <= terrain_coords[1] < self.height_samples.shape[1] - 2
                    ), f"Scene {scene_idx}: Object {object_name} is outside the valid range of height samples (y-axis)"

                    # Assert that the object is in the designated spawn area
                    assert (
                        self.terrain.tot_cols
                        - self.terrain.border
                        - self.terrain.object_playground_cols
                        <= terrain_coords[1]
                        < self.terrain.tot_cols - self.terrain.border
                    ), f"Scene {scene_idx}: Object {object_name} is not in the designated spawn area"

                    # Assert that the terrain is not "flat" at the object's location
                    assert not (
                        self.terrain.flat_field_raw[
                            terrain_coords[0], terrain_coords[1]
                        ]
                        == 0
                    ), f"Scene {scene_idx}: Object {object_name} is placed on flat terrain"

                    terrain_height = self.get_ground_heights(
                        global_object_position[:2].unsqueeze(0)
                    ).item()
                    global_object_position[2] = (
                        terrain_height + initial_object_pose.translations[0, 2]
                    )

                    object_pose.p = gymapi.Vec3(
                        global_object_position[0],
                        global_object_position[1],
                        global_object_position[2],
                    )
                    object_pose.r = gymapi.Quat(
                        initial_object_pose.rotations[0, 0],
                        initial_object_pose.rotations[0, 1],
                        initial_object_pose.rotations[0, 2],
                        initial_object_pose.rotations[0, 3],
                    )

                    object_category = object_spawn_info.object_path.split("/")[-2]

                    self.object_id_to_scene_id.append(scene_idx)

                    object_target_position = self.object_target_positions[
                        self.scene_lib.object_path_to_id[object_spawn_info.object_path]
                    ]
                    self.object_target_position.append(
                        object_target_position + global_object_position
                    )
                    self.spawned_object_names.append(
                        object_category + "_" + object_name
                    )

                    object_handle = self.gym.create_actor(
                        env_ptr, object_asset, object_pose, object_name, -1, 0
                    )
                    self.object_handles.append(object_handle)
                    object_idx = self.gym.get_actor_index(
                        env_ptr, object_handle, gymapi.DOMAIN_SIM
                    )
                    self.object_indices.append(object_idx)

                    # Extract the object name from the full path
                    object_name = os.path.splitext(
                        os.path.basename(object_spawn_info.object_path)
                    )[0]

                    # Ensure the .obj file exists
                    obj_path = object_spawn_info.object_path.replace(".urdf", ".obj")
                    stl_path = object_spawn_info.object_path.replace(".urdf", ".stl")
                    ply_path = object_spawn_info.object_path.replace(".urdf", ".ply")

                    if (
                        os.path.exists(obj_path)
                        or os.path.exists(stl_path)
                        or os.path.exists(ply_path)
                    ):
                        if os.path.exists(obj_path):
                            mesh_path = obj_path
                        elif os.path.exists(stl_path):
                            mesh_path = stl_path
                        else:
                            mesh_path = ply_path
                        mesh = as_mesh(trimesh.load_mesh(mesh_path))
                        w_x, w_y, w_z, m_x, m_y, m_z = compute_bounding_box(mesh)
                        # Sample points evenly from the mesh surface

                    elif object_spawn_info.object_path.endswith(".urdf"):
                        import xml.etree.ElementTree as ET

                        tree = ET.parse(object_spawn_info.object_path)
                        root = tree.getroot()
                        link = root.find("link")
                        has_size = False
                        if link is not None:
                            collision = link.find("collision")
                            if collision is not None:
                                geometry = collision.find("geometry")
                                if geometry is not None:
                                    box = geometry.find("box")
                                    if box is not None:
                                        size = box.get("size")

                                        w_x, w_y, w_z = map(float, size.split())
                                        m_x = -w_x / 2
                                        m_y = -w_y / 2
                                        m_z = -w_z / 2
                                        has_size = True
                        assert (
                            has_size
                        ), f"URDF {object_spawn_info.object_path} must provide size parameters."
                    else:
                        raise FileNotFoundError(
                            f"Object file not found: {obj_path}, {stl_path}, or valid URDF"
                        )

                    min_x = m_x
                    max_x = min_x + w_x
                    min_y = m_y
                    max_y = min_y + w_y
                    min_z = m_z
                    max_z = min_z + w_z

                    self.object_dims.append(
                        torch.tensor(
                            [min_x, max_x, min_y, max_y, min_z, max_z],
                            device=self.device,
                            dtype=torch.float,
                        )
                    )

                    # Use offsets from spawn_info for object_root_states_offsets
                    translation_offset = self.scene_lib.object_translation_offsets[
                        object_id
                    ]
                    rotation_offset = self.scene_lib.object_rotation_offsets[object_id]

                    self.object_root_states_offsets.append(
                        torch.cat(
                            [
                                translation_offset,
                                rotation_offset,
                                torch.tensor(
                                    [self.config.object_types.index(object_category)],
                                    device=self.device,
                                    dtype=torch.float,
                                ),
                            ]
                        )
                    )

                    scale = 2.0
                    heightmap_path = osp.join(
                        os.path.dirname(object_spawn_info.object_path),
                        f"{object_name}_{scale}_{self.terrain.config.horizontal_scale}.pt",
                    )
                    if osp.exists(heightmap_path):
                        heightmap = torch.load(heightmap_path)
                    else:
                        print(
                            "Creating object heightmap for object {} at scale {}".format(
                                object_name, scale
                            )
                        )
                        heightmap = torch.tensor(
                            get_object_heightmap(
                                mesh,
                                dim_x=int(
                                    np.ceil(
                                        w_x
                                        / (self.terrain.config.horizontal_scale / scale)
                                    )
                                ),
                                dim_y=int(
                                    np.ceil(
                                        w_y
                                        / (self.terrain.config.horizontal_scale / scale)
                                    )
                                ),
                            ),
                            dtype=torch.float,
                        )
                        torch.save(heightmap, heightmap_path)

                    heightmap = heightmap.to(self.device)

                    # 1. Create a grid for the object in global coordinates --> each cell has the global coordinates of the center of that cell.
                    # 2. Do the same for the heightmap.
                    # 3. Go cell by cell in the heightmap, where the object resides.
                    # 3.1. Find the appropriate cells in the object grid, and perform bilinear interpolation to get the height at that point.

                    object_min_coords = [
                        (
                            scene_offset[0]
                            + initial_object_pose.translations[0, 0]
                            + m_x
                        ).item(),
                        (
                            scene_offset[1]
                            + initial_object_pose.translations[0, 1]
                            + m_y
                        ).item(),
                    ]
                    object_max_coords = [
                        object_min_coords[0] + w_x,
                        object_min_coords[1] + w_y,
                    ]
                    object_min_cell_idx = [
                        int(np.floor(coord / self.terrain.config.horizontal_scale))
                        for coord in object_min_coords
                    ]
                    object_max_cell_idx = [
                        int(np.ceil(coord / self.terrain.config.horizontal_scale))
                        for coord in object_max_coords
                    ]

                    for x in range(
                        object_min_cell_idx[0] - 1, object_max_cell_idx[0] + 1
                    ):
                        for y in range(
                            object_min_cell_idx[1] - 1, object_max_cell_idx[1] + 1
                        ):
                            # get coordinates in object-relative frame, remove object offset
                            object_coords = [
                                x * self.terrain.config.horizontal_scale,
                                y * self.terrain.config.horizontal_scale,
                            ]
                            object_coords = [
                                object_coords[0]
                                - (
                                    scene_offset[0]
                                    + initial_object_pose.translations[0, 0]
                                ).item(),
                                object_coords[1]
                                - (
                                    scene_offset[1]
                                    + initial_object_pose.translations[0, 1]
                                ).item(),
                            ]
                            object_coords = [
                                object_coords[0] - m_x,
                                object_coords[1] - m_y,
                            ]

                            object_floor_idx = [
                                int(
                                    np.floor(
                                        object_coords[0]
                                        / (self.terrain.config.horizontal_scale / scale)
                                    )
                                ),
                                int(
                                    np.floor(
                                        object_coords[1]
                                        / (self.terrain.config.horizontal_scale / scale)
                                    )
                                ),
                            ]

                            # TODO: For now, pick max height since there's some issue with billinear due to discretization size

                            # perform billinear interpolation, if out of bounds interpolate with 0
                            x1 = object_floor_idx[0]
                            x2 = x1 + 1
                            y1 = object_floor_idx[1]
                            y2 = y1 + 1
                            # xm = object_coords[0] / (
                            #     self.terrain.config.horizontal_scale / scale
                            # )
                            # ym = object_coords[1] / (
                            #     self.terrain.config.horizontal_scale / scale
                            # )

                            x1y1 = (
                                heightmap[x1, y1]
                                if 0 <= x1 < heightmap.shape[0]
                                and 0 <= y1 < heightmap.shape[1]
                                else 0
                            )
                            x2y1 = (
                                heightmap[x2, y1]
                                if 0 <= x2 < heightmap.shape[0]
                                and 0 <= y1 < heightmap.shape[1]
                                else 0
                            )
                            x1y2 = (
                                heightmap[x1, y2]
                                if 0 <= x1 < heightmap.shape[0]
                                and 0 <= y2 < heightmap.shape[1]
                                else 0
                            )
                            x2y2 = (
                                heightmap[x2, y2]
                                if 0 <= x2 < heightmap.shape[0]
                                and 0 <= y2 < heightmap.shape[1]
                                else 0
                            )

                            # height_point = (x2 - xm) * (y2 - ym) * x1y1 + (xm - x1) * (y2 - ym) * x2y1 + (x2 - xm) * (ym - y1) * x1y2 + (xm - x1) * (ym - y1) * x2y2
                            height_point = max(x1y1, x2y1, x1y2, x2y2)

                            self.height_samples[x, y] += height_point

    ###############################################################
    # Getters
    ###############################################################
    def get_humanoid_root_states(self):
        return self.humanoid_root_states[..., :7].clone()

    def get_num_actors_per_env(self):
        num_actors = (
            self.root_states.shape[0] - self.total_num_objects
        ) // self.num_envs
        return num_actors

    def get_body_id(self, body_name):
        return self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.humanoid_handles[0], body_name
        )

    def get_body_positions(self):
        return self.rigid_body_pos.clone()

    def get_bodies_contact_buf(self):
        return self.contact_forces.clone()

    def get_dof_offsets(self):
        return self.dof_offsets

    def get_bodies_state(self):
        body_pos = self.rigid_body_pos.clone()
        body_rot = self.rigid_body_rot.clone()
        body_vel = self.rigid_body_vel.clone()
        body_ang_vel = self.rigid_body_ang_vel.clone()

        return_dict = EasyDict(
            {
                "body_pos": body_pos,
                "body_rot": body_rot,
                "body_vel": body_vel,
                "body_ang_vel": body_ang_vel,
            }
        )
        return return_dict

    def get_dof_state(self):
        return self.dof_pos.clone(), self.dof_vel.clone()

    def get_humanoid_root_velocities(self):
        return self.humanoid_root_states[:, 7:10].clone()

    ###############################################################
    # Environment step logic
    ###############################################################
    def apply_pd_control(self):
        pd_tar = self.action_to_pd_targets(self.actions)
        pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
        self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)

    def apply_motor_forces(self):
        torques = self.compute_torques(self.actions)
        torques_tensor = gymtorch.unwrap_tensor(torques)
        self.gym.set_dof_actuation_force_tensor(self.sim, torques_tensor)

    def refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.reset_happened:
            env_ids = self.reset_ref_env_ids
            self.humanoid_root_states[env_ids, 0:3] = self.reset_states["root_pos"]
            self.humanoid_root_states[env_ids, 3:7] = self.reset_states["root_rot"]
            self.humanoid_root_states[env_ids, 7:10] = self.reset_states["root_vel"]
            self.humanoid_root_states[env_ids, 10:13] = self.reset_states[
                "root_ang_vel"
            ]

            self.dof_pos[env_ids] = self.reset_states["dof_pos"]
            self.dof_vel[env_ids] = self.reset_states["dof_vel"]

            self.rigid_body_pos[env_ids] = self.reset_states["rb_pos"]
            self.rigid_body_rot[env_ids] = self.reset_states["rb_rot"]
            self.rigid_body_vel[env_ids] = self.reset_states["rb_vel"]
            self.rigid_body_ang_vel[env_ids] = self.reset_states["rb_ang_vel"]

            if self.object_reset_states is not None:
                object_ids = self.reset_ref_object_ids
                self.object_root_states[object_ids, 0:3] = self.object_reset_states[
                    "position"
                ]
                self.object_root_states[object_ids, 3:7] = self.object_reset_states[
                    "rotation"
                ]
                self.object_root_states[object_ids, 7:10] = self.object_reset_states[
                    "velocity"
                ]
                self.object_root_states[object_ids, 10:13] = self.object_reset_states[
                    "angular_velocity"
                ]
                self.object_reset_states = None
            self.reset_happened = False

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def post_physics_step(self):
        self.refresh_sim_tensors()

        super().post_physics_step()

        if self.export_video:
            self.camera_step()

        if self.export_motion:
            self.store_motion_data()

    ###############################################################
    # Handle Resets
    ###############################################################
    def reset_envs(self, env_ids):
        super().reset_envs(env_ids)

        if len(env_ids) > 0:
            self.reset_actors(env_ids)
            self.reset_env_tensors(env_ids)
            self.refresh_sim_tensors()
            self.compute_observations(env_ids)

    def reset_env_tensors(self, env_ids, object_ids=None):
        super().reset_env_tensors(env_ids)

        actor_ids = self.humanoid_actor_ids[env_ids]
        set_root_state_ids = actor_ids

        if object_ids is not None and len(object_ids) > 0:

            object_actor_ids = self.object_indices[object_ids]
            set_root_state_ids = torch.cat(
                [set_root_state_ids, object_actor_ids], dim=0
            )

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(set_root_state_ids),
            len(set_root_state_ids),
        )

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids),
        )
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_pos.contiguous()),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids),
        )

    def reset_default(self, env_ids):
        # Reset all humanoid states to default zero-state
        root_pos = self.initial_humanoid_root_states[env_ids, 0:3].clone()
        root_rot = self.initial_humanoid_root_states[env_ids, 3:7].clone()
        root_vel = self.initial_humanoid_root_states[env_ids, 7:10].clone()
        root_ang_vel = self.initial_humanoid_root_states[env_ids, 10:13].clone()
        dof_pos = self.initial_dof_pos[env_ids].clone()
        dof_vel = self.initial_dof_vel[env_ids].clone()
        rb_pos = self.initial_rigid_body_pos[env_ids].clone()
        rb_rot = self.initial_rigid_body_rot[env_ids].clone()
        rb_vel = self.initial_rigid_body_vel[env_ids].clone()
        rb_ang_vel = self.initial_rigid_body_ang_vel[env_ids].clone()

        # Adjust root position
        root_pos[:, :2] = 0
        root_pos += self.get_envs_respawn_position(env_ids)

        self.set_env_state(
            env_ids,
            root_pos,
            root_rot,
            dof_pos,
            root_vel,
            root_ang_vel,
            dof_vel,
            rb_pos,
            rb_rot,
            rb_vel,
            rb_ang_vel,
        )

    def set_env_state(
        self,
        env_ids,
        root_pos,
        root_rot,
        dof_pos,
        root_vel,
        root_ang_vel,
        dof_vel,
        rb_pos,
        rb_rot,
        rb_vel,
        rb_ang_vel,
    ):
        """
        Set the state of specified environments.

        This method updates the state of the humanoid and rigid bodies for the given environment IDs.
        It sets the root position, rotation, velocity, and angular velocity of the humanoid,
        as well as the degrees of freedom (DOF) positions and velocities, and the rigid body
        positions, rotations, velocities, and angular velocities.

        Args:
            env_ids (Tensor): The IDs of the environments to update.
            root_pos (Tensor): Root positions for the humanoids.
            root_rot (Tensor): Root rotations for the humanoids.
            dof_pos (Tensor): DOF positions for the humanoids.
            root_vel (Tensor): Root velocities for the humanoids.
            root_ang_vel (Tensor): Root angular velocities for the humanoids.
            dof_vel (Tensor): DOF velocities for the humanoids.
            rb_pos (Tensor): Rigid body positions.
            rb_rot (Tensor): Rigid body rotations.
            rb_vel (Tensor): Rigid body velocities.
            rb_ang_vel (Tensor): Rigid body angular velocities.

        Note:
            This method also stores the reset states in a dictionary for potential future use.
        """
        # Update humanoid root states
        self.humanoid_root_states[env_ids, 0:3] = root_pos
        self.humanoid_root_states[env_ids, 3:7] = root_rot
        self.humanoid_root_states[env_ids, 7:10] = root_vel
        self.humanoid_root_states[env_ids, 10:13] = root_ang_vel

        # Update DOF states
        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel

        # Update rigid body states
        self.rigid_body_pos[env_ids] = rb_pos
        self.rigid_body_rot[env_ids] = rb_rot
        self.rigid_body_vel[env_ids] = rb_vel
        self.rigid_body_ang_vel[env_ids] = rb_ang_vel

        # Store reset states
        self.reset_states = {
            "root_pos": root_pos.clone(),
            "root_rot": root_rot.clone(),
            "root_vel": root_vel.clone(),
            "root_ang_vel": root_ang_vel.clone(),
            "dof_pos": dof_pos.clone(),
            "dof_vel": dof_vel.clone(),
            "rb_pos": rb_pos.clone(),
            "rb_rot": rb_rot.clone(),
            "rb_vel": rb_vel.clone(),
            "rb_ang_vel": rb_ang_vel.clone(),
        }

    def set_object_state(self, object_ids, positions, rotations):
        """
        Set the state of specified objects in the environment.

        This method updates the root state of objects identified by object_ids. It calculates
        the appropriate position based on the scene and terrain, and sets the rotation, velocity,
        and angular velocity for each object.

        Args:
            object_ids (Tensor): The IDs of the objects to update.
            positions (Tensor): The new positions for the objects, relative to their respective scenes.
            rotations (Tensor): The new rotations for the objects.

        Note:
            - The input positions are relative to the scene, not global coordinates.
            - This method adjusts for terrain height and scene position to set global object positions.
            - Reset states of the objects are stored for potential future use.
        """
        # Get scene information for the objects
        scene_id = self.object_id_to_scene_id[object_ids]
        scene_position = self.scene_position[scene_id]

        # Calculate terrain height at object positions
        # Note: positions are relative to the scene, so we add scene_position for global coordinates
        terrain_height = self.get_ground_heights((positions + scene_position)[..., :2])

        # Update object root states
        # Convert scene-relative positions to global positions
        self.object_root_states[object_ids, 0:3] = positions + scene_position
        self.object_root_states[object_ids, 2] += terrain_height.view(-1)
        self.object_root_states[object_ids, 3:7] = rotations
        self.object_root_states[object_ids, 7:10] = 0  # Set velocity to zero
        self.object_root_states[object_ids, 10:13] = 0  # Set angular velocity to zero

        # Store reset states for objects (in global coordinates)
        self.object_reset_states = {
            "position": self.object_root_states[object_ids, 0:3].clone(),
            "rotation": self.object_root_states[object_ids, 3:7].clone(),
            "velocity": self.object_root_states[object_ids, 7:10].clone(),
            "angular_velocity": self.object_root_states[object_ids, 10:13].clone(),
        }

    ###############################################################
    # Helpers
    ###############################################################
    def set_char_color(self, col, env_ids):
        for env_id in env_ids:
            env_ptr = self.envs[env_id]
            handle = self.humanoid_handles[env_id]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr,
                    handle,
                    j,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(col[0], col[1], col[2]),
                )

    def setup_character_props(self):
        self.dof_body_ids = self.config.robot.dof_body_ids
        self.dof_offsets = self.config.robot.dof_offsets
        self.dof_obs_size = self.config.robot.dof_obs_size
        self.num_obs = self.config.robot.self_obs_max_coords
        self.num_act = self.config.robot.number_of_actions

    def render(self):
        if self.viewer:
            self.update_camera()
            self.gym.clear_lines(self.viewer)
            self.draw_object_bounding_boxes()

        super().render()

    def draw_object_bounding_boxes(self):
        """
        Draw bounding boxes and direction indicators for objects in the scene.

        This method visualizes the bounding boxes of objects and their direction
        vectors in the simulation environment. It uses the viewer to add lines
        representing the bounding boxes and direction indicators.

        The method performs the following steps:
        1. Check if there are any objects to draw.
        2. Prepare colors for bounding box lines and direction indicators.
        3. Calculate and draw bounding box lines.
        4. Calculate and draw direction indicators for each object.

        Note: This method should only be called when the viewer is available.
        """
        if len(self.scene_position) == 0:
            return

        # Colors for visualization
        bounding_box_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Red
        direction_indicator_color = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Green

        # Draw bounding boxes
        bounding_boxes = self.object_id_to_object_bounding_box(None)
        box_vertices = self._calculate_bounding_box_vertices(bounding_boxes)
        self._draw_lines(box_vertices, bounding_box_color)

        # Draw direction indicators
        object_positions, object_directions = self._calculate_object_directions()
        direction_vertices = (
            torch.cat([object_positions, object_directions], dim=-1).cpu().numpy()
        )
        self._draw_lines(direction_vertices, direction_indicator_color)

    def _calculate_bounding_box_vertices(self, bounding_boxes):
        """
        Calculate vertices for drawing bounding box lines.

        Args:
            bounding_boxes (torch.Tensor): Tensor containing bounding box coordinates.

        Returns:
            numpy.ndarray: Array of vertices for drawing lines.
        """
        vertex_indices = [
            0,
            1,
            1,
            2,
            2,
            3,
            3,
            0,
            4,
            5,
            5,
            6,
            6,
            7,
            7,
            4,
            0,
            4,
            1,
            5,
            2,
            6,
            3,
            7,
        ]
        return (
            torch.cat([bounding_boxes[:, i, :] for i in vertex_indices], dim=-1)
            .cpu()
            .numpy()
        )

    def _calculate_object_directions(self):
        """
        Calculate object positions and direction vectors.

        Returns:
            tuple: (object_positions, object_directions) as torch.Tensor
        """
        object_positions = self.object_root_states[..., :3].clone()
        scene_id = self.object_id_to_scene_id[:]

        object_rotations = rotations.quat_mul(
            self.object_root_states_offsets[..., 3:7],
            self.object_root_states[..., 3:7],
            self.w_last,
        )
        direction_vectors = torch.zeros_like(self.object_root_states[..., :3])
        direction_vectors[..., 0] = 1
        rotated_directions = torch_utils.quat_rotate(
            object_rotations, direction_vectors, self.w_last
        )

        return object_positions, object_positions + rotated_directions

    def _draw_lines(self, vertices, color):
        """
        Draw lines in the viewer using the provided vertices and color.

        Args:
            vertices (numpy.ndarray): Array of vertex coordinates.
            color (numpy.ndarray): RGB color values for the lines.
        """
        env_ptr = self.envs[0]
        vertices = vertices.reshape(-1, 6)
        self.gym.add_lines(self.viewer, env_ptr, vertices.shape[0], vertices, color)

    def build_body_ids_tensor(self, body_names):
        body_ids = []

        for body_name in body_names:
            body_id = self.body_name_to_index(body_name)
            assert body_id != -1
            body_ids.append(body_id)

        body_ids = torch_utils.to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def body_name_to_index(self, body_name):
        return self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.humanoid_handles[0], body_name
        )

    ###############################################################
    # Camera logic
    ###############################################################
    def init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.cam_prev_char_pos = self.humanoid_root_states[0, 0:3].cpu().numpy()

        cam_pos = gymapi.Vec3(
            self.cam_prev_char_pos[0],
            self.cam_prev_char_pos[1] - 3.0,
            self.cam_prev_char_pos[2] + 0.4,
        )
        cam_target = gymapi.Vec3(
            self.cam_prev_char_pos[0],
            self.cam_prev_char_pos[1],
            self.cam_prev_char_pos[2] + 0.2,
        )
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def update_camera(self):
        """
        Update the camera position and target to follow the humanoid character.

        This method performs the following steps:
        1. Refresh the actor root state tensor to get the latest character position.
        2. Get the current camera position and calculate its offset from the previous character position.
        3. Calculate the new camera target and position based on the character's current position.
        4. Update the viewer camera to look at the new target from the new position.
        5. Store the current character position for the next update.
        """
        # Refresh actor state and get current character root position
        self.gym.refresh_actor_root_state_tensor(self.sim)
        current_char_pos = self.humanoid_root_states[0, 0:3].cpu().numpy()

        # Get current camera transform and position
        current_cam_transform = self.gym.get_viewer_camera_transform(self.viewer, None)
        current_cam_pos = np.array(
            [
                current_cam_transform.p.x,
                current_cam_transform.p.y,
                current_cam_transform.p.z,
            ]
        )

        # Calculate camera offset from previous character position
        cam_offset = current_cam_pos - self.cam_prev_char_pos

        # Calculate new camera target (slightly above character)
        new_cam_target = gymapi.Vec3(
            current_char_pos[0], current_char_pos[1], current_char_pos[2] + 0.2
        )

        # Calculate new camera position (maintaining relative offset)
        new_cam_pos = gymapi.Vec3(
            current_char_pos[0] + cam_offset[0],
            current_char_pos[1] + cam_offset[1],
            current_char_pos[2] + cam_offset[2],
        )

        # Update viewer camera
        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        # Store current character position for next update
        self.cam_prev_char_pos[:] = current_char_pos

    def setup_cameras(self):
        self.cameras = []

        camera_config = self.config.camera

        camera_props = gymapi.CameraProperties()
        camera_props.width = camera_config.width
        camera_props.height = camera_config.height
        camera_offset = gymapi.Vec3(*camera_config.pos)
        camera_props.enable_tensors = True

        pitch = np.deg2rad(camera_config.pitch_deg)
        roll = np.deg2rad(camera_config.roll_deg)
        yaw = np.deg2rad(camera_config.yaw_deg)
        camera_rotation = gymapi.Quat.from_euler_zyx(roll, pitch, yaw)
        transform = gymapi.Transform(camera_offset, camera_rotation)

        follow = (
            gymapi.FOLLOW_TRANSFORM
            if camera_config["rotate_with_agent"]
            else gymapi.FOLLOW_POSITION
        )

        for env, han in zip(self.envs, self.humanoid_handles):
            body_handle = self.gym.get_actor_rigid_body_handle(env, han, 0)
            camera_handle = self.gym.create_camera_sensor(env, camera_props)
            self.gym.attach_camera_to_body(
                camera_handle, env, body_handle, transform, follow
            )
            self.cameras.append(camera_handle)

        self.frames = [[] for _ in range(len(self.cameras))]
        self.cpu_frames = [[] for _ in range(len(self.cameras))]
        self.max_gpu_frames = 1000

    def camera_step(self):
        if self.config.record_viewer:
            viewer_record_dir = self.config.viewer_record_dir
            if not os.path.exists(viewer_record_dir):
                os.makedirs(viewer_record_dir)
            self.gym.write_viewer_image_to_file(
                self.viewer, viewer_record_dir + "/%04d.png" % len(self.frames[0])
            )

        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        for i, han in enumerate(self.cameras):
            camera_tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim, self.envs[i], han, gymapi.IMAGE_COLOR
            )
            torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)
            torch_camera_tensor = torch_camera_tensor[:, :, :3]
            self.frames[i].append(torch_camera_tensor.clone().cpu().numpy())

        self.gym.end_access_image_tensors(self.sim)

    def store_motion_data(self):
        """
        Store the current motion data of the humanoid.

        This method captures and records various aspects of the humanoid's motion,
        including root position, global rotation, and rigid body states.
        The data is stored in the motion_recording dictionary for later use or analysis.
        """
        # Capture root position
        root_position = self.humanoid_root_states[..., 0:3].clone()

        # Process body rotation
        body_rotation = self.rigid_body_rot.clone()
        negative_w_mask = body_rotation[..., -1] < 0
        body_rotation[negative_w_mask] = -body_rotation[negative_w_mask]

        # Prepare motion data dictionary
        current_motion_data = {
            "root_pos": root_position.cpu(),
            "global_rot": body_rotation.cpu(),
            "rigid_body_rot": self.rigid_body_rot.clone().cpu(),
            "rigid_body_pos": self.rigid_body_pos.clone().cpu(),
        }

        # Store motion data
        for data_key, data_value in current_motion_data.items():
            if data_key not in self.motion_recording:
                self.motion_recording[data_key] = []
            self.motion_recording[data_key].append(data_value)

    def apply_sideways_force_to_feet(self):
        """
        Apply a sideways force to the feet of the humanoid.

        This method creates force and torque tensors for all rigid bodies in the simulation,
        including the humanoid and any objects. It then applies a constant sideways force
        to specific body parts of the humanoid (presumably the feet).

        The forces are applied in the global coordinate system of the simulation environment.
        """
        # Initialize force and torque tensors for all rigid bodies
        total_bodies = self.rigid_body_state.shape[0] + self.total_num_objects
        forces = torch.zeros((total_bodies, 3), device=self.device, dtype=torch.float)
        torques = torch.zeros((total_bodies, 3), device=self.device, dtype=torch.float)

        # Define the magnitude of the applied force
        FORCE = -3500

        # Apply downward force to specific body parts (feet)
        num_humanoids = self.rigid_body_state.shape[0] // self.num_bodies
        for humanoid_index in range(num_humanoids):
            left_foot_index = humanoid_index * self.num_bodies + 3
            right_foot_index = humanoid_index * self.num_bodies + 7
            forces[left_foot_index, :] = FORCE
            forces[right_foot_index, :] = FORCE

        # Apply the forces and torques to the simulation
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(forces),
            gymtorch.unwrap_tensor(torques),
            gymapi.ENV_SPACE,
        )

    ###############################################################
    # Helpers
    ###############################################################
    def instantiate_motion_lib(self):
        spawned_scenes = None
        if self.scene_lib is not None:
            spawned_scenes = self.scene_lib.get_scene_ids()
        motion_lib: MotionLib = instantiate(
            self.config.motion_lib,
            dof_body_ids=self.dof_body_ids,
            dof_offsets=self.dof_offsets,
            key_body_ids=self.key_body_ids,
            device=self.device,
            spawned_scene_ids=spawned_scenes,
            skeleton_tree=None,
        )
        return motion_lib
