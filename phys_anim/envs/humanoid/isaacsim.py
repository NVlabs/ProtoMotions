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

from pathlib import Path
import torch
import numpy as np
from easydict import EasyDict
from rich.progress import Progress
from hydra.utils import instantiate
from isaac_utils import rotations

from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView, XFormPrim, XFormPrimView
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.physx.scripts import physicsUtils
from pxr import UsdGeom, Vt, UsdPhysics, PhysxSchema, Gf

from phys_anim.envs.humanoid.common import BaseHumanoid
from phys_anim.utils.file_utils import load_yaml
from phys_anim.envs.humanoid.humanoid_utils import build_pd_action_offset_scale
from phys_anim.envs.base_interface.isaacsim import SimBaseInterface
from phys_anim.envs.base_interface.isaacsim_utils.perspective_viewer import (
    PerspectiveViewer,
)
from phys_anim.envs.base_interface.isaacsim_utils.usd_utils import (
    add_terrain_to_stage,
)
from phys_anim.utils.motion_lib import MotionLib


class Humanoid(BaseHumanoid, SimBaseInterface):
    def __init__(self, config, device: torch.device) -> None:
        # Although quaternion definition in isaacsim is w_first, we convert the API calls and not MotionLib.
        self.w_last = True
        self.config = config
        self.cameras_config = self.config.cameras
        self.device = device

        self.dt: float = (
            self.config.simulator.sim.control_freq_inv
            * 1.0
            / self.config.simulator.sim.fps
        )
        self.perspective_view = None
        self.cameras = None

        super().__init__(config, device)

        self.build_termination_heights()

        # Allows the agent to disable resets temporarily.
        self.disable_reset = False
        self.num_humanoid_cams = min(self.num_envs, 5)

        # Call at the end to enable base_interface classes to generate the required base_interface tensors.
        self.body_names = self.joints_humanoids_view.body_names

        self.on_environment_ready()

    ###############################################################
    # Set up IsaacSim environment
    ###############################################################
    def set_up_scene(self) -> None:
        if self.terrain is not None:
            self.add_terrain()

        self.create_robot()
        super().set_up_scene()

        root_prim = self.bodies_names[0]

        self.joints_humanoids_view = ArticulationView(
            prim_paths_expr=f"/World/envs/.*/{self.config.robot.asset.robot_type}/bodies/{root_prim}",
            name="humanoid_articulation_view",
            reset_xform_properties=False,
        )

        self.bodies_humanoids_view = RigidPrimView(
            prim_paths_expr=f"/World/envs/.*/{self.config.robot.asset.robot_type}/bodies/.*",
            name="humanoid_rigid_prim_view",
            reset_xform_properties=False,
            track_contact_forces=True,
            prepare_contact_sensors=True,
        )

        if (
            self.config.scene_lib is not None
            and self.scene_lib.total_spawned_scenes > 0
        ):
            self.build_object_playground()
            self.objects_view = XFormPrimView(
                f"{self.default_object_path}/object_.*/.*",
            )

        self._world.scene.add(self.joints_humanoids_view)
        self._world.scene.add(self.bodies_humanoids_view)

    def build_object_playground(self):
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

                    object_name = object_spawn_info.object_path.split("/")[-1].split(
                        "."
                    )[0]
                    file_extension = object_spawn_info.object_path.split("/")[-1].split(
                        "."
                    )[-1]

                    assert file_extension in [
                        "usd",
                        "usda",
                    ], f"Object asset [{object_spawn_info.object_path}] must be a USD file"

                    main_dir_path = (
                        f"{os.path.dirname(os.path.abspath(__file__))}/../../../"
                    )
                    asset_path = Path(
                        os.path.join(main_dir_path, object_spawn_info.object_path)
                    ).resolve()

                    prim_path = (
                        self.default_object_path + f"/object_{self.total_num_objects}"
                    )
                    self.total_num_objects += 1

                    add_reference_to_stage(str(asset_path), prim_path)

                    stage = get_current_stage()
                    prim = stage.GetPrimAtPath(prim_path)

                    object_options = object_spawn_info.object_options
                    if object_options.vhacd_enabled:
                        # Apply the CollisionAPI to the prim
                        collision_api = UsdPhysics.CollisionAPI.Apply(prim)

                        # Apply the MeshCollisionAPI to the prim
                        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)

                        # Set the collision approximation to convex decomposition
                        mesh_collision_api.CreateApproximationAttr().Set(
                            "convexDecomposition"
                        )

                        # Apply the PhysxConvexDecompositionCollisionAPI to the prim
                        convex_api = (
                            PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                        )

                        if "vhacd_params" in object_options:
                            if "max_convex_hulls" in object_options.vhacd_params:
                                convex_api.CreateMaxConvexHullsAttr().Set(
                                    object_options.vhacd_params.max_convex_hulls
                                )
                            if "max_num_vertices_per_ch" in object_options.vhacd_params:
                                convex_api.CreateHullVertexLimitAttr().Set(
                                    object_options.vhacd_params.max_num_vertices_per_ch
                                )
                            if "resolution" in object_options.vhacd_params:
                                convex_api.CreateVoxelResolutionAttr().Set(
                                    object_options.vhacd_params.resolution
                                )
                    if (
                        "fix_base_link" in object_options
                        and object_options.fix_base_link
                    ):
                        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
                        rigid_body_api.CreateRigidBodyEnabledAttr().Set(False)

                        # Create a light orange color (you can adjust these values to get the desired shade)
                        light_orange = Gf.Vec3f(1.0, 0.6, 0.2)  # RGB values
                        # Create a VtArray with the color
                        color_array = Vt.Vec3fArray([light_orange])

                        # Get the UsdGeom.Gprim from the prim (this works for any geometric prim like Mesh, Cube, Sphere, etc.)
                        geom = UsdGeom.Gprim(prim)

                        # Create the displayColor attribute if it doesn't exist
                        if not geom.GetDisplayColorAttr():
                            geom.CreateDisplayColorAttr()

                        # Set the displayColor attribute
                        geom.GetDisplayColorAttr().Set(color_array)

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

                    obj = XFormPrim(
                        prim_path=prim_path,
                        name=object_name,
                        translation=global_object_position,
                        orientation=rotations.xyzw_to_wxyz(
                            initial_object_pose.rotations[0, :4]
                        ),
                        scale=torch.tensor([1.0, 1.0, 1.0], device=self.device),
                    )

                    object_category = object_spawn_info.object_path.split("/")[-2]

                    self.object_id_to_scene_id.append(scene_idx)

                    object_info = self.scene_lib.object_spawn_list[
                        self.scene_lib.object_path_to_id[object_spawn_info.object_path]
                    ]

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
                    object_target_position = torch.tensor(
                        target_position, device=self.device, dtype=torch.float
                    ).view(-1)

                    self.object_target_position.append(
                        object_target_position + global_object_position
                    )
                    self.spawned_object_names.append(
                        object_category + "_" + object_name
                    )

                    # Extract the object name from the full path
                    object_name = os.path.splitext(
                        os.path.basename(object_spawn_info.object_path)
                    )[0]

                    # Ensure the .obj file exists
                    obj_path = object_spawn_info.object_path.replace(
                        ".usda", ".usd"
                    ).replace(".usd", ".obj")
                    stl_path = object_spawn_info.object_path.replace(
                        ".usda", ".usd"
                    ).replace(".usd", ".stl")
                    ply_path = object_spawn_info.object_path.replace(
                        ".usda", ".usd"
                    ).replace(".usd", ".ply")

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

    def _debug_print_all_prims(self):
        stage = self._world.stage
        print("\nAll prims in the scene:")
        self._debug_print_prim_recursive(stage.GetPseudoRoot(), 0)

    def _debug_print_prim_recursive(self, prim, indent_level):
        indent = "  " * indent_level
        print(f"{indent}{prim.GetPath()}")
        for child in prim.GetChildren():
            self._debug_print_prim_recursive(child, indent_level + 1)

    def _debug_joints(self):
        print(self.joints_humanoids_view.dof_names)
        print(self.joints_humanoids_view.body_names)

    def create_robot(self):
        main_dir_path = f"{os.path.dirname(os.path.abspath(__file__))}/../../../"
        asset_path = Path(
            os.path.join(
                main_dir_path,
                self.config.robot.asset.asset_root,
                self.config.robot.asset.asset_file_name,
            )
        ).resolve()

        prim_path = (
            self.default_zero_env_path + f"/{self.config.robot.asset.robot_type}"
        )
        file_extension = asset_path.suffix.lower()

        if file_extension == ".xml":  # MJCF
            raise NotImplementedError
        elif file_extension == ".urdf":
            raise NotImplementedError
        elif "usd" in file_extension:
            add_reference_to_stage(str(asset_path), prim_path)

        humanoid = Robot(
            prim_path=prim_path,
            name="Humanoid",
        )

        self.sim_config.apply_articulation_settings(
            "Humanoid",
            get_prim_at_path(humanoid.prim_path),
            self.sim_config.parse_actor_config("Humanoid"),
        )

    def add_terrain(self):
        stage = get_current_stage()
        vertices = self.terrain.vertices
        triangles = self.terrain.triangles
        position = torch.tensor([0.0, 0.0, 0.0])
        add_terrain_to_stage(
            stage=stage, vertices=vertices, triangles=triangles, position=position
        )

    ###############################################################
    # Getters
    ###############################################################
    def get_body_id(self, body_name):
        return self.bodies_names.index(body_name)

    def get_observations(self):
        return self.obs_buf

    def get_bodies_state(self):
        isaacsim_bodies_positions, isaacsim_bodies_rotations = (
            self.bodies_humanoids_view.get_world_poses()
        )
        isaacsim_velocities = self.bodies_humanoids_view.get_velocities()
        isaacsim_bodies_velocities = isaacsim_velocities[..., :3]
        isaacsim_bodies_ang_velocities = isaacsim_velocities[..., 3:]
        isaacsim_bodies_positions = isaacsim_bodies_positions.view(
            self.num_envs, self.num_bodies, 3
        )
        isaacsim_bodies_rotations = isaacsim_bodies_rotations.view(
            self.num_envs, self.num_bodies, 4
        )
        isaacsim_bodies_velocities = isaacsim_bodies_velocities.view(
            self.num_envs, self.num_bodies, 3
        )
        isaacsim_bodies_ang_velocities = isaacsim_bodies_ang_velocities.view(
            self.num_envs, self.num_bodies, 3
        )

        isaacsim_bodies_rotations = rotations.wxyz_to_xyzw(isaacsim_bodies_rotations)

        if self.rigid_body_indices_isaac_sim_to_gym is not None:
            bodies_positions = isaacsim_bodies_positions[
                :, self.rigid_body_indices_isaac_sim_to_gym
            ]
            bodies_rotations = isaacsim_bodies_rotations[
                :, self.rigid_body_indices_isaac_sim_to_gym
            ]
            bodies_velocities = isaacsim_bodies_velocities[
                :, self.rigid_body_indices_isaac_sim_to_gym
            ]
            bodies_ang_velocities = isaacsim_bodies_ang_velocities[
                :, self.rigid_body_indices_isaac_sim_to_gym
            ]
        else:
            bodies_positions = isaacsim_bodies_positions
            bodies_rotations = isaacsim_bodies_rotations
            bodies_velocities = isaacsim_bodies_velocities
            bodies_ang_velocities = isaacsim_bodies_ang_velocities

        return_dict = EasyDict(
            {
                "body_pos": bodies_positions,
                "body_rot": bodies_rotations,
                "body_vel": bodies_velocities,
                "body_ang_vel": bodies_ang_velocities,
            }
        )

        return return_dict

    def get_dof_forces(self):
        isaacsim_dof_forces = self.joints_humanoids_view.get_measured_joint_efforts()
        if self.dof_body_indices_isaac_sim_to_gym is not None:
            dof_forces = isaacsim_dof_forces[
                :, self.rigid_body_indices_isaac_sim_to_gym
            ]
        else:
            dof_forces = isaacsim_dof_forces
        return dof_forces

    def get_dof_state(self) -> tuple:
        isaacsim_dof_pos = self.joints_humanoids_view.get_joint_positions()
        isaacsim_dof_vel = self.joints_humanoids_view.get_joint_velocities()
        if self.dof_body_indices_isaac_sim_to_gym is not None:
            dof_pos = isaacsim_dof_pos[:, self.dof_offset_indices_isaac_sim_to_gym]
            dof_vel = isaacsim_dof_vel[:, self.dof_offset_indices_isaac_sim_to_gym]
        else:
            dof_pos = isaacsim_dof_pos
            dof_vel = isaacsim_dof_vel
        return dof_pos, dof_vel

    def get_body_positions(self):
        isaacsim_rb_pos = self.bodies_humanoids_view.get_world_poses()[0].view(
            self.num_envs, self.num_bodies, 3
        )
        if self.rigid_body_indices_isaac_sim_to_gym is not None:
            rb_pos = isaacsim_rb_pos[:, self.rigid_body_indices_isaac_sim_to_gym]
        else:
            rb_pos = isaacsim_rb_pos
        return rb_pos

    def get_bodies_contact_buf(self):
        isaacsim_rb_contacts = self.bodies_humanoids_view.get_net_contact_forces().view(
            self.num_envs, self.num_bodies, 3
        )
        if self.rigid_body_indices_isaac_sim_to_gym is not None:
            rb_contacts = isaacsim_rb_contacts[
                :, self.rigid_body_indices_isaac_sim_to_gym
            ]
        else:
            rb_contacts = isaacsim_rb_contacts
        return rb_contacts

    def get_humanoid_root_states(self):
        root_pos, isaacsim_root_rot = self.joints_humanoids_view.get_world_poses()
        root_rot = rotations.wxyz_to_xyzw(isaacsim_root_rot)
        return torch.cat((root_pos, root_rot), dim=-1)

    def get_object_root_states(self):
        root_pos, isaacsim_root_rot = self.objects_view.get_world_poses()
        root_rot = rotations.wxyz_to_xyzw(isaacsim_root_rot)
        return torch.cat((root_pos, root_rot), dim=-1)

    def get_num_actors_per_env(self):
        root_pos, root_rot = self.joints_humanoids_view.get_world_poses()
        return root_pos.shape[0] // self.num_envs

    ###############################################################
    # Environment step logic
    ###############################################################
    def post_reset(self):
        """CALLED ONCE AFTER SCENE LOAD"""
        dof_limits = self.joints_humanoids_view.get_dof_limits()
        self.dof_limits_lower = dof_limits[0, :, 0].to(self.device)
        self.dof_limits_upper = dof_limits[0, :, 1].to(self.device)

        # Build pd_actions scales
        self._pd_action_offset, self._pd_action_scale = build_pd_action_offset_scale(
            self.dof_offsets,
            self.dof_limits_lower,
            self.dof_limits_upper,
            self.device,
        )
        self.initial_root_pos, self.initial_root_rot = (
            self.joints_humanoids_view.get_world_poses()
        )
        self.initial_root_vel = torch.zeros(
            (len(self.initial_root_pos), 6), device=self.device
        )
        self.initial_dof_pos = torch.zeros_like(
            self.joints_humanoids_view.get_joint_positions(),
            device=self.device,
            dtype=torch.float32,
        )
        self.initial_dof_vel = torch.zeros_like(
            self.joints_humanoids_view.get_joint_velocities(),
            device=self.device,
            dtype=torch.float32,
        )

        self.init_done = True

    def world_running(self):
        return self._world.is_playing()

    def apply_pd_control(self):
        isaacsim_actions = self.actions[:, self.dof_offset_indices_isaac_gym_to_sim]
        pd_tar = self.action_to_pd_targets(isaacsim_actions)
        self.joints_humanoids_view.set_joint_position_targets(pd_tar)

    def apply_motor_forces(self):
        if not self.config.sync_motion:
            raise NotImplementedError

    ###############################################################
    # Handle Resets
    ###############################################################
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
        root_rot = rotations.xyzw_to_wxyz(root_rot)
        rb_rot = rotations.xyzw_to_wxyz(rb_rot)

        if self.dof_body_indices_isaac_gym_to_sim is not None:
            dof_pos = dof_pos[:, self.dof_offset_indices_isaac_gym_to_sim]
            dof_vel = dof_vel[:, self.dof_offset_indices_isaac_gym_to_sim]

        if self.rigid_body_indices_isaac_gym_to_sim is not None:
            rb_pos = rb_pos[:, self.rigid_body_indices_isaac_gym_to_sim]
            rb_rot = rb_rot[:, self.rigid_body_indices_isaac_gym_to_sim]
            rb_vel = rb_vel[:, self.rigid_body_indices_isaac_gym_to_sim]
            rb_ang_vel = rb_ang_vel[:, self.rigid_body_indices_isaac_gym_to_sim]

        root_vel = torch.cat([root_vel, root_ang_vel], -1)

        self.joints_humanoids_view.set_world_poses(root_pos, root_rot, indices=env_ids)
        self.joints_humanoids_view.set_velocities(root_vel, indices=env_ids)
        self.joints_humanoids_view.set_joint_positions(dof_pos, indices=env_ids)
        self.joints_humanoids_view.set_joint_position_targets(dof_pos, indices=env_ids)
        self.joints_humanoids_view.set_joint_velocities(dof_vel, indices=env_ids)

    def reset_envs(self, env_ids):
        if len(env_ids) > 0:
            self.reset_actors(env_ids)
            self.reset_env_tensors(env_ids)
            self.set_char_color(np.array([0.65, 0.1, 1.0]), env_ids)

            self.compute_observations(env_ids)

    def reset_default(self, env_ids):
        respawn_position = self.get_envs_respawn_position(env_ids)
        initial_root_pos = self.initial_root_pos[env_ids].clone()
        initial_root_pos[..., :2] = 0
        initial_root_pos[..., :3] += respawn_position

        self.joints_humanoids_view.set_world_poses(
            initial_root_pos[env_ids], self.initial_root_rot[env_ids], indices=env_ids
        )
        self.joints_humanoids_view.set_velocities(
            self.initial_root_vel[env_ids], indices=env_ids
        )
        self.joints_humanoids_view.set_joint_positions(
            self.initial_dof_pos[env_ids], indices=env_ids
        )
        self.joints_humanoids_view.set_joint_velocities(
            self.initial_dof_vel[env_ids], indices=env_ids
        )

    ###############################################################
    # Helpers
    ###############################################################
    def setup_character_props(self):
        self.bodies_names = self.config.robot.dfs_body_names
        self.num_bodies = self.config.robot.num_bodies
        self.key_body_ids = torch.tensor(
            [
                self.bodies_names.index(key_body_name)
                for key_body_name in self.config.robot.key_bodies
            ],
            device=self.device,
            dtype=torch.long,
        )
        self.dfs_key_body_ids = torch.tensor(
            [
                self.config.robot.dfs_body_names.index(key_body_name)
                for key_body_name in self.config.robot.key_bodies
            ],
            device=self.device,
            dtype=torch.long,
        )
        self.non_termination_contact_body_ids = torch.tensor(
            [
                # self.bodies_names.index(non_termination_contact_body_name)
                self.config.robot.dfs_body_names.index(
                    non_termination_contact_body_name
                )
                for non_termination_contact_body_name in self.config.robot.non_termination_contact_bodies
            ],
            device=self.device,
            dtype=torch.long,
        )

        self.contact_body_ids = torch.tensor(
            [
                # self.bodies_names.index(non_termination_contact_body_name)
                self.config.robot.dfs_body_names.index(contact_body_name)
                for contact_body_name in self.config.robot.contact_bodies
            ],
            device=self.device,
            dtype=torch.long,
        )

        dfs_dof_body_names = [
            self.config.robot.dfs_body_names[dof_body_id]
            for dof_body_id in self.config.robot.dfs_dof_body_ids
        ]
        self.dof_body_ids = [
            i
            for i in range(len(self.config.robot.bfs_body_names))
            if self.config.robot.bfs_body_names[i] in dfs_dof_body_names
        ]

        self.dof_offsets = []
        previous_dof_name = "null"
        for dof_offset, dof_name in enumerate(self.config.robot.dfs_dof_names):
            if dof_name[:-2] != previous_dof_name:  # remove the "_x/y/z"
                previous_dof_name = dof_name[:-2]
                self.dof_offsets.append(dof_offset)
        self.dof_offsets.append(len(self.config.robot.dfs_dof_names))

        self.dfs_dof_offsets = []
        previous_dof_name = "null"
        for dof_offset, dof_name in enumerate(self.config.robot.dfs_dof_names):
            if dof_name[:-2] != previous_dof_name:  # remove the "_x/y/z"
                previous_dof_name = dof_name[:-2]
                self.dfs_dof_offsets.append(dof_offset)
        self.dfs_dof_offsets.append(len(self.config.robot.dfs_dof_names))

        self.bfs_dof_offsets = []
        previous_dof_name = "null"
        for dof_offset, dof_name in enumerate(self.config.robot.bfs_dof_names):
            if dof_name[:-2] != previous_dof_name:  # remove the "_x/y/z"
                previous_dof_name = dof_name[:-2]
                self.bfs_dof_offsets.append(dof_offset)
        self.bfs_dof_offsets.append(len(self.config.robot.bfs_dof_names))

        self.dof_obs_size = self.config.robot.dof_obs_size
        self.num_act = self.config.robot.number_of_actions

        self.dof_offset_indices_isaac_gym_to_sim = torch.tensor(
            [
                self.config.robot.dfs_dof_names.index(dof_name)
                for dof_name in self.config.robot.bfs_dof_names
            ],
            dtype=torch.long,
            device=self.device,
        )
        # For some reason IsaacSim returns the rigid body in dfs ordering.
        self.rigid_body_indices_isaac_gym_to_sim = None

        self.dof_offset_indices_isaac_sim_to_gym = torch.tensor(
            [
                self.config.robot.bfs_dof_names.index(dof_name)
                for dof_name in self.config.robot.dfs_dof_names
            ],
            dtype=torch.long,
            device=self.device,
        )
        # For some reason IsaacSim returns the rigid body in dfs ordering.
        self.rigid_body_indices_isaac_sim_to_gym = None

        #     torch.tensor(
        #     [
        #         self.config.robot.dfs_body_names.index(body_name)
        #         for body_name in self.config.robot.bfs_body_names
        #     ],
        #     dtype=torch.long,
        #     device=self.device,
        # )

        bfs_dof_body_names = [
            self.config.robot.bfs_body_names[dof_body_id]
            for dof_body_id in self.dof_body_ids
        ]
        dof_body_indices_isaac_gym_to_sim = []
        for i in range(self.config.robot.num_bodies):
            if self.config.robot.bfs_body_names[i] in bfs_dof_body_names:
                dof_body_indices_isaac_gym_to_sim.append(
                    dfs_dof_body_names.index(self.config.robot.bfs_body_names[i])
                )
            else:
                dof_body_indices_isaac_gym_to_sim.append(i)
        self.dof_body_indices_isaac_gym_to_sim = torch.tensor(
            dof_body_indices_isaac_gym_to_sim,
            dtype=torch.long,
            device=self.device,
        )

        dfs_dof_body_ids = [
            i
            for i in range(len(self.config.robot.dfs_body_names))
            if self.config.robot.dfs_body_names[i] in dfs_dof_body_names
        ]
        dfs_dof_body_names = [
            self.config.robot.dfs_body_names[dof_body_id]
            for dof_body_id in dfs_dof_body_ids
        ]
        dof_body_indices_isaac_sim_to_gym = []
        for i in range(self.config.robot.num_bodies):
            if self.config.robot.dfs_body_names[i] in dfs_dof_body_names:
                dof_body_indices_isaac_sim_to_gym.append(
                    bfs_dof_body_names.index(self.config.robot.dfs_body_names[i])
                )
            else:
                dof_body_indices_isaac_sim_to_gym.append(i)
        self.dof_body_indices_isaac_sim_to_gym = torch.tensor(
            dof_body_indices_isaac_sim_to_gym,
            dtype=torch.long,
            device=self.device,
        )

    def set_char_color(self, col, env_ids):
        if self.init_done and not self.headless:
            stage = get_current_stage()
            all_prim_paths = self.bodies_humanoids_view._prim_paths
            color_array = Vt.Vec3fArray.FromNumpy(col)
            for i in range(len(all_prim_paths)):
                if i // self.num_bodies in env_ids:
                    UsdGeom.Gprim.Get(
                        stage, all_prim_paths[i]
                    ).CreateDisplayColorAttr().Set(value=color_array)

    def render(self):
        if not self.headless and self.init_done:
            from omni.isaac.sensor import Camera

            if self.perspective_view is None:
                self.perspective_view = PerspectiveViewer()
                self.init_camera()

                self.cameras = []
                for i in range(self.num_humanoid_cams):
                    self.cameras.append(
                        Camera(
                            prim_path="/World/envs/env_"
                            + str(i)
                            + "/camera_human_"
                            + str(i),
                            position=np.array([0.0, 0.0, 25.0]),
                            frequency=15,
                            resolution=(256, 256),
                            orientation=rot_utils.euler_angles_to_quats(
                                np.array([0, 90, 0]), degrees=True
                            ),
                        )
                    )
                    self.cameras[i].initialize()
                for camera_config in self.config.cameras:
                    self.cameras.append(
                        Camera(
                            prim_path="/World/envs/env_"
                            + str(i)
                            + "/camera_"
                            + camera_config.name,
                            position=np.array(camera_config.position),
                            frequency=camera_config.frequency,
                            resolution=tuple(camera_config.resolution),
                            orientation=rot_utils.euler_angles_to_quats(
                                np.array(camera_config.orientation), degrees=True
                            ),
                        )
                    )
                    self.cameras[-1].initialize()
            else:
                self.update_camera()

    def init_camera(self):
        self.cam_prev_char_pos = (
            self.get_humanoid_root_states()[: self.num_humanoid_cams, :3].cpu().numpy()
        )
        pos = self.cam_prev_char_pos[0, :] + np.array([0, -5, 1])
        self.perspective_view.set_camera_view(
            pos, self.cam_prev_char_pos[0, :] + np.array([0, 0, 0.2])
        )

    def update_camera(self):
        char_root_pos = (
            self.get_humanoid_root_states()[: self.num_humanoid_cams, :3].cpu().numpy()
        )
        cam_pos = np.array(self.perspective_view.get_camera_state())
        cam_delta = cam_pos - self.cam_prev_char_pos[0]

        ego_char_root_pos = char_root_pos[0, :]
        new_cam_target = np.array(
            [ego_char_root_pos[0], ego_char_root_pos[1], ego_char_root_pos[2] + 0.2]
        )
        new_cam_pos = np.array(
            [
                ego_char_root_pos[0] + cam_delta[0],
                ego_char_root_pos[1] + cam_delta[1],
                ego_char_root_pos[2] + cam_delta[2],
            ]
        )
        self.perspective_view.set_camera_view(new_cam_pos, new_cam_target)

        root_pos = self.get_humanoid_root_states()[: self.num_humanoid_cams, :]
        import math

        for cam_idx in range(self.num_humanoid_cams):
            camera_pos = self.cameras[cam_idx].get_world_pose()[0]
            cam_delta = camera_pos - torch.tensor(
                self.cam_prev_char_pos[cam_idx, :], device=self.device
            )
            X = root_pos[cam_idx, 0] - camera_pos[0]
            Y = root_pos[cam_idx, 1] - camera_pos[1]
            Z = root_pos[cam_idx, 2] - camera_pos[2]
            pitch = math.atan2(math.sqrt(X**2 + Y**2), Z) * 180 / math.pi - 90
            yaw = math.atan2(Y, X) * 180 / math.pi
            looking_to_root = torch.tensor(
                rot_utils.euler_angles_to_quats(
                    np.array([0, pitch, yaw]), degrees=True
                ),
                device=root_pos.device,
            )
            self.cameras[cam_idx].set_world_pose(
                root_pos[cam_idx, :3] + cam_delta, looking_to_root
            )

        self.cam_prev_char_pos[:] = char_root_pos

    def output_motion(self):
        # TODO: add code to record states
        raise NotImplementedError

    def instantiate_motion_lib(self):
        spawned_scenes = None
        if self.scene_lib is not None:
            spawned_scenes = self.scene_lib.get_scene_ids()
        motion_lib: MotionLib = instantiate(
            self.config.motion_lib,
            dof_body_ids=self.config.robot.dfs_dof_body_ids,
            dof_offsets=self.bfs_dof_offsets,
            key_body_ids=self.dfs_key_body_ids.cpu().numpy(),
            device=self.device,
            spawned_scene_ids=spawned_scenes,
            skeleton_tree=None,
        )
        return motion_lib
