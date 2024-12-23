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
from isaac_utils import rotations

import xml.etree.ElementTree as ET

from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.stage import get_current_stage
from omni.physx.scripts import physicsUtils, utils
from pxr import UsdShade, UsdGeom, Vt, UsdPhysics, PhysxSchema, Gf

from phys_anim.envs.humanoid.common import BaseHumanoid
from phys_anim.utils.file_utils import load_yaml
from phys_anim.envs.base_interface.isaaclab import SimBaseInterface
from phys_anim.envs.base_interface.isaaclab_utils.usd_utils import (
    add_terrain_to_stage,
)


class Humanoid(BaseHumanoid, SimBaseInterface):
    def __init__(self, config, device: torch.device, simulation_app):
        super().__init__(config, device, simulation_app)
        self.perspective_view = None

        self.build_termination_heights()

        # Allows the agent to disable resets temporarily.
        self.disable_reset = False

        # Call at the end to enable base_interface classes to generate the required base_interface tensors.
        self.body_names = self.robot.data.body_names

        self.on_environment_ready()

    ###############################################################
    # Set up IsaacSim environment
    ###############################################################
    def set_up_scene(self) -> None:
        self.add_terrain()

        super().set_up_scene()

        self.objects_view = None
        if (
            self.config.scene_lib is not None
            and self.scene_lib.total_spawned_scenes > 0
        ):
            self.build_object_playground()
            self.objects_view = RigidPrimView(
                prim_paths_expr=f"{self.default_object_path}/object_.*",
                name="object_rigid_prim_view",
                reset_xform_properties=False,
                track_contact_forces=True,
                prepare_contact_sensors=True,
            )

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

                height_at_scene_origin = self.terrain_obs_cb.get_ground_heights(
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
                    object_options = object_spawn_info.object_options

                    object_name = object_spawn_info.object_path.split("/")[-1].split(
                        "."
                    )[0]
                    file_extension = object_spawn_info.object_path.split("/")[-1].split(
                        "."
                    )[-1]

                    assert file_extension in [
                        "usd",
                        "usda",
                        "urdf",
                    ], f"Object asset [{object_spawn_info.object_path}] must be a USD file"

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
                        0
                        <= terrain_coords[0]
                        < self.terrain_obs_cb.height_samples.shape[0] - 2
                    ), f"Scene {scene_idx}: Object {object_name} is outside the valid range of height samples (x-axis)"
                    assert (
                        0
                        <= terrain_coords[1]
                        < self.terrain_obs_cb.height_samples.shape[1] - 2
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

                    terrain_height = self.terrain_obs_cb.get_ground_heights(
                        global_object_position[:2].unsqueeze(0)
                    ).item()
                    global_object_position[2] = (
                        terrain_height + initial_object_pose.translations[0, 2]
                    )

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

                    rotation = rotations.xyzw_to_wxyz(
                        initial_object_pose.rotations[0, :4]
                    )

                    if file_extension == "urdf":
                        # We currently only support cubes for URDFs

                        # Parse the URDF file
                        tree = ET.parse(asset_path)
                        root = tree.getroot()

                        # Get the box dimensions from the collision geometry
                        link = root.find("link")
                        collision = link.find("collision")
                        geometry = collision.find("geometry")
                        box = geometry.find("box")
                        size = box.get("size").split(" ")

                        obj = UsdGeom.Cube.Define(get_current_stage(), prim_path)
                        obj.AddTranslateOp()
                        obj.AddScaleOp()
                        obj.AddOrientOp()
                        obj.GetPrim().GetAttribute("xformOp:scale").Set(
                            Gf.Vec3f(
                                float(size[0]) / 2,
                                float(size[1]) / 2,
                                float(size[2]) / 2,
                            )
                        )
                    else:
                        obj = UsdGeom.Xform.Define(get_current_stage(), prim_path)
                        obj.GetPrim().GetReferences().AddReference(str(asset_path))

                    obj.GetPrim().GetAttribute("xformOp:translate").Set(
                        Gf.Vec3f(
                            global_object_position[0].item(),
                            global_object_position[1].item(),
                            global_object_position[2].item(),
                        )
                    )
                    obj.GetPrim().GetAttribute("xformOp:orient").Set(
                        Gf.Quatf(
                            rotation[0].item(),
                            rotation[1].item(),
                            rotation[2].item(),
                            rotation[3].item(),
                        )
                    )

                    physx_rb_api = PhysxSchema.PhysxRigidBodyAPI(obj.GetPrim())
                    if not physx_rb_api:
                        physx_rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(
                            obj.GetPrim()
                        )
                    physx_rb_api.GetEnableGyroscopicForcesAttr().Set(True)
                    physx_rb_api.GetMaxDepenetrationVelocityAttr().Set(10.0)
                    physx_rb_api.GetSolverPositionIterationCountAttr().Set(8)
                    physx_rb_api.GetSolverVelocityIterationCountAttr().Set(0)

                    if object_options.vhacd_enabled:
                        # Apply the CollisionAPI to the prim
                        UsdPhysics.CollisionAPI.Apply(obj.GetPrim())

                        # Apply the MeshCollisionAPI to the prim
                        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(
                            obj.GetPrim()
                        )

                        # Set the collision approximation to convex decomposition
                        mesh_collision_api.CreateApproximationAttr().Set(
                            "convexDecomposition"
                        )

                        # Apply the PhysxConvexDecompositionCollisionAPI to the prim
                        convex_api = (
                            PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(
                                obj.GetPrim()
                            )
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

                        # # change SDF params and check if recooking happens.
                        # meshCollision = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(obj.GetPrim())
                        # meshCollision.CreateSdfResolutionAttr().Set(1000)
                        # # need to explicitly change to triangle mesh to start SDF cooking.
                        # meshcollisionAPI = UsdPhysics.MeshCollisionAPI.Apply(obj.GetPrim())
                        # meshcollisionAPI.CreateApproximationAttr().Set("sdf")

                        # PhysxSchema.PhysxCollisionAPI.Apply(obj.GetPrim())
                    if (
                        "fix_base_link" in object_options
                        and object_options.fix_base_link
                    ):
                        rigid_body_api = UsdPhysics.RigidBodyAPI.Get(
                            get_current_stage(), obj.GetPrim().GetPrimPath()
                        )
                        rigid_body_api.CreateKinematicEnabledAttr().Set(True)

                        # Create a light orange color
                        light_orange = Gf.Vec3f(1.0, 0.6, 0.2)  # RGB values
                        # Create a VtArray with the color
                        color_array = Vt.Vec3fArray([light_orange])
                    else:
                        obj_material_path = f"/World/Physics_Materials/objects/object_{self.total_num_objects - 1}"
                        utils.addRigidBodyMaterial(
                            get_current_stage(),
                            obj_material_path,
                            density=object_options.density,
                            staticFriction=10.0,
                            dynamicFriction=10.0,
                            restitution=0.0,
                        )

                        get_current_stage().GetPrimAtPath(
                            str(obj.GetPrim().GetPrimPath())
                        ).SetInstanceable(
                            False
                        )  # This is required to be able to edit physics material
                        physicsUtils.add_physics_material_to_prim(
                            get_current_stage(),
                            get_current_stage().GetPrimAtPath(
                                obj.GetPrim().GetPrimPath()
                            ),
                            obj_material_path,
                        )

                        # Create a green color
                        dark_green = Gf.Vec3f(0.2, 0.7, 0.3)  # RGB values
                        # Create a VtArray with the color
                        color_array = Vt.Vec3fArray([dark_green])

                    # Get the UsdGeom.Gprim from the prim (this works for any geometric prim like Mesh, Cube, Sphere, etc.)
                    geom = UsdGeom.Gprim(obj.GetPrim())

                    # Create the displayColor attribute if it doesn't exist
                    # if not geom.GetDisplayColorAttr():
                    geom.CreateDisplayColorAttr()

                    # Set the displayColor attribute
                    geom.GetDisplayColorAttr().Set(color_array)

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
                    obj_path = (
                        object_spawn_info.object_path.replace(".usda", ".usd")
                        .replace(".usd", ".obj")
                        .replace(".urdf", ".obj")
                    )
                    stl_path = (
                        object_spawn_info.object_path.replace(".usda", ".usd")
                        .replace(".usd", ".stl")
                        .replace(".urdf", ".stl")
                    )
                    ply_path = (
                        object_spawn_info.object_path.replace(".usda", ".usd")
                        .replace(".usd", ".ply")
                        .replace(".urdf", ".ply")
                    )

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
                        point_cloud = trimesh.sample.sample_surface_even(
                            mesh, self.config.point_cloud_obs.num_pointcloud_samples
                        )[0]
                        if (
                            point_cloud.shape[0]
                            < self.config.point_cloud_obs.num_pointcloud_samples
                        ):
                            # Even spacing uses rejection sampling, as a result it may return less points than requested
                            # we add the extra points by randomly sampling the mesh surface again
                            missing_points = (
                                self.config.point_cloud_obs.num_pointcloud_samples
                                - point_cloud.shape[0]
                            )
                            extra_points = trimesh.sample.sample_surface(
                                mesh, missing_points
                            )[0]
                            point_cloud = np.concatenate(
                                [point_cloud, extra_points], axis=0
                            )
                    elif object_spawn_info.object_path.endswith(".urdf"):
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
                        # Generate point cloud for URDF box
                        points_per_dim = int(
                            np.ceil(
                                self.config.point_cloud_obs.num_pointcloud_samples
                                ** (1 / 3)
                            )
                        )
                        x = torch.linspace(m_x, m_x + w_x, points_per_dim)
                        y = torch.linspace(m_y, m_y + w_y, points_per_dim)
                        z = torch.linspace(m_z, m_z + w_z, points_per_dim)
                        xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
                        point_cloud = torch.stack(
                            [xx.flatten(), yy.flatten(), zz.flatten()], dim=1
                        )

                        # Randomly select exactly 64 points if we have more
                        if (
                            point_cloud.shape[0]
                            > self.config.point_cloud_obs.num_pointcloud_samples
                        ):
                            indices = torch.randperm(point_cloud.shape[0])[
                                : self.config.point_cloud_obs.num_pointcloud_samples
                            ]
                            point_cloud = point_cloud[indices]
                        elif (
                            point_cloud.shape[0]
                            < self.config.point_cloud_obs.num_pointcloud_samples
                        ):
                            # If we have less than 64 points, duplicate some randomly
                            extra_indices = torch.randint(
                                0,
                                point_cloud.shape[0],
                                (
                                    self.config.point_cloud_obs.num_pointcloud_samples
                                    - point_cloud.shape[0],
                                ),
                            )
                            point_cloud = torch.cat(
                                [point_cloud, point_cloud[extra_indices]], dim=0
                            )
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
                    tensor_pointcloud = torch.tensor(
                        point_cloud, device=self.device, dtype=torch.float
                    )
                    assert tensor_pointcloud.shape == (
                        self.config.point_cloud_obs.num_pointcloud_samples,
                        3,
                    ), f"Expected shape ({self.config.point_cloud_obs.num_pointcloud_samples}, 3), got {tensor_pointcloud.shape}"
                    if self.config.point_cloud_obs.enabled:
                        self.object_obs_cb.add_initial_object_pointcloud(
                            tensor_pointcloud
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
                            ]
                        )
                    )

                    if object_spawn_info.object_options.participates_in_heightmap:
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
                                            / (
                                                self.terrain.config.horizontal_scale
                                                / scale
                                            )
                                        )
                                    ),
                                    dim_y=int(
                                        np.ceil(
                                            w_y
                                            / (
                                                self.terrain.config.horizontal_scale
                                                / scale
                                            )
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
                                            / (
                                                self.terrain.config.horizontal_scale
                                                / scale
                                            )
                                        )
                                    ),
                                    int(
                                        np.floor(
                                            object_coords[1]
                                            / (
                                                self.terrain.config.horizontal_scale
                                                / scale
                                            )
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

                                self.terrain_obs_cb.height_samples[x, y] += height_point

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
        print(self.robot.data.joint_names)
        print(self.robot.data.body_names)

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
    def get_bodies_state(self):
        isaacsim_bodies_positions = self.robot.data.body_pos_w.clone()
        isaacsim_bodies_rotations = self.robot.data.body_quat_w.clone()
        isaacsim_bodies_velocities = self.robot.data.body_lin_vel_w.clone()
        isaacsim_bodies_ang_velocities = self.robot.data.body_ang_vel_w.clone()

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

        bodies_positions = isaacsim_bodies_positions[
            :, self.body_isaac_sim_to_gym
        ]
        bodies_rotations = isaacsim_bodies_rotations[
            :, self.body_isaac_sim_to_gym
        ]
        bodies_velocities = isaacsim_bodies_velocities[
            :, self.body_isaac_sim_to_gym
        ]
        bodies_ang_velocities = isaacsim_bodies_ang_velocities[
            :, self.body_isaac_sim_to_gym
        ]

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
        isaacsim_dof_forces = self.robot.data.applied_torque.clone()

        dof_forces = isaacsim_dof_forces[:, self.dof_isaac_sim_to_gym]
        return dof_forces

    def get_dof_state(self) -> tuple:
        isaacsim_dof_pos = self.robot.data.joint_pos.clone()
        isaacsim_dof_vel = self.robot.data.joint_vel.clone()

        dof_pos = isaacsim_dof_pos[:, self.dof_isaac_sim_to_gym]
        dof_vel = isaacsim_dof_vel[:, self.dof_isaac_sim_to_gym]
        return dof_pos, dof_vel

    def get_body_positions(self):
        isaacsim_rb_pos = self.robot.data.body_pos_w.clone().view(
            self.num_envs, self.num_bodies, 3
        )
        rb_pos = isaacsim_rb_pos[:, self.body_isaac_sim_to_gym]
        return rb_pos

    def get_bodies_contact_buf(self):
        if self.contact_sensor.data.force_matrix_w is not None:
            isaacsim_rb_contacts = self.contact_sensor.data.force_matrix_w.view(
                self.num_envs, self.num_bodies, -1, 3
            )
            contacts = isaacsim_rb_contacts.sum(dim=2)
        else:
            contacts = self.contact_sensor.data.net_forces_w.clone().view(
                self.num_envs, self.num_bodies, 3
            )
        rb_contacts = contacts[
            :, self.contact_sensor_isaac_sim_to_gym
        ]
        return rb_contacts

    def get_object_contact_buf(self):
        raise NotImplementedError

    def get_bodies_forces_buf(self):
        raise NotImplementedError

    def get_object_forces_buf(self):
        object_contact_forces = self.objects_view.get_net_contact_forces()
        return object_contact_forces

    def get_humanoid_root_states(self):
        root_pos = self.robot.data.root_pos_w.clone()
        root_rot = rotations.wxyz_to_xyzw(self.robot.data.root_quat_w.clone())
        return torch.cat((root_pos, root_rot), dim=-1)

    def get_object_root_states(self):
        root_pos, isaacsim_root_rot = self.objects_view.get_world_poses()
        root_rot = rotations.wxyz_to_xyzw(isaacsim_root_rot)
        return torch.cat((root_pos, root_rot), dim=-1)

    def get_num_actors_per_env(self):
        root_pos = self.robot.data.root_pos_w
        return root_pos.shape[0] // self.num_envs

    ###############################################################
    # Environment step logic
    ###############################################################
    def on_environment_ready(self):
        """CALLED ONCE AFTER SCENE LOAD"""
        dof_limits = self.robot.data.joint_limits.clone()
        self.dof_limits_lower = dof_limits[0, :, 0].to(self.device)
        self.dof_limits_upper = dof_limits[0, :, 1].to(self.device)

        self.initial_root_pos = self.robot.data.root_pos_w.clone()
        self.initial_root_rot = self.robot.data.root_quat_w.clone()
        self.initial_root_vel = torch.zeros(
            (len(self.initial_root_pos), 6), device=self.device
        )
        self.initial_dof_pos = torch.zeros_like(
            self.robot.data.joint_pos, device=self.device, dtype=torch.float32
        )
        self.initial_dof_vel = torch.zeros_like(
            self.robot.data.joint_vel, device=self.device, dtype=torch.float32
        )

        super().on_environment_ready()

    def world_running(self):
        return self.simulation_app.is_running()

    def apply_pd_control(self):
        isaacsim_actions = self.actions[:, self.dof_isaac_gym_to_sim]
        pd_tar = self.action_to_pd_targets(isaacsim_actions)
        self.robot.set_joint_position_target(pd_tar, joint_ids=None)

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

        root_rot = rotations.xyzw_to_wxyz(root_rot)

        dof_pos = dof_pos[:, self.dof_isaac_gym_to_sim]
        dof_vel = dof_vel[:, self.dof_isaac_gym_to_sim]

        init_root_state = torch.cat(
            [root_pos, root_rot, root_vel, root_ang_vel], dim=-1
        )
        self.robot.write_root_state_to_sim(init_root_state, env_ids)
        self.robot.write_joint_state_to_sim(dof_pos, dof_vel, None, env_ids)

    def set_object_state(self, object_ids, obj_pos, obj_rot):
        """
        Set the state of specified objects in the environment.

        This method updates the root state of objects identified by object_ids. It calculates
        the appropriate position based on the scene and terrain, and sets the rotation, velocity,
        and angular velocity for each object.

        Args:
            object_ids (Tensor): The IDs of the objects to update.
            obj_pos (Tensor): The new positions for the objects, relative to their respective scenes.
            obj_rot (Tensor): The new rotations for the objects.

        Note:
            - The input positions are relative to the scene, not global coordinates.
            - This method adjusts for terrain height and scene position to set global object positions.
        """
        # Get scene information for the objects
        scene_id = self.object_id_to_scene_id[object_ids]
        scene_position = self.scene_position[scene_id]

        # Calculate terrain height at object positions
        # Note: positions are relative to the scene, so we add scene_position for global coordinates
        terrain_height = self.terrain_obs_cb.get_ground_heights(
            (obj_pos + scene_position)[..., :2]
        )

        # Update object root states
        root_pos = obj_pos + scene_position
        root_pos[..., 2] += terrain_height.view(-1)
        sim_obj_rot = rotations.xyzw_to_wxyz(obj_rot)
        self.objects_view.set_world_poses(root_pos, sim_obj_rot, object_ids)
        velocities = torch.zeros(
            len(object_ids), 6, device=self.device, dtype=torch.float
        )
        self.objects_view.set_velocities(velocities, object_ids)

        self.object_reset_states = {
            "position": root_pos,
            "rotation": obj_rot,
            "velocity": velocities[..., :3],
            "angular_velocity": velocities[..., 3:],
        }

    def reset_envs(self, env_ids):
        if len(env_ids) > 0:
            self.reset_actors(env_ids)
            self.reset_env_tensors(env_ids)
            # self.set_char_color(np.array([0.65, 0.1, 1.0]), env_ids)
            self.compute_observations(env_ids)
            super().reset_envs(env_ids)

    def reset_default(self, env_ids):
        respawn_position = self.get_envs_respawn_position(env_ids)
        initial_root_pos = self.initial_root_pos[env_ids].clone()
        initial_root_pos[..., :2] = 0
        initial_root_pos[..., :3] += respawn_position

        self.set_env_state(
            env_ids,
            initial_root_pos,
            self.initial_root_rot[env_ids],
            self.initial_dof_pos[env_ids],
            self.initial_root_vel[env_ids],
            self.initial_root_ang_vel[env_ids],
            self.initial_dof_vel[env_ids],
            self.initial_rb_pos[env_ids],
            self.initial_rb_rot[env_ids],
            self.initial_rb_vel[env_ids],
            self.initial_rb_ang_vel[env_ids],
        )

    ###############################################################
    # Helpers
    ###############################################################
    def setup_character_props(self):
        super().setup_character_props()
        isaacsim_dof_names = self.robot.data.joint_names
        isaacsim_body_names = self.robot.data.body_names
        isaacsim_contact_sensor_body_names = self.contact_sensor.body_names

        self.body_isaac_gym_to_sim = torch.tensor(
            [
                self.config.robot.isaacgym_body_names.index(body_name)
                for body_name in isaacsim_body_names
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.body_isaac_sim_to_gym = torch.tensor(
            [
                isaacsim_body_names.index(body_name)
                for body_name in self.config.robot.isaacgym_body_names
            ],
            dtype=torch.long,
            device=self.device,
        )

        self.contact_sensor_isaac_sim_to_gym = torch.tensor(
            [
                self.config.robot.isaacgym_body_names.index(body_name)
                for body_name in isaacsim_contact_sensor_body_names
            ],
            dtype=torch.long,
            device=self.device,
        )

        self.dof_isaac_gym_to_sim = torch.tensor(
            [
                self.config.robot.isaacgym_dof_names.index(dof_name)
                for dof_name in isaacsim_dof_names
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.dof_isaac_sim_to_gym = torch.tensor(
            [
                isaacsim_dof_names.index(dof_name)
                for dof_name in self.config.robot.isaacgym_dof_names
            ],
            dtype=torch.long,
            device=self.device,
        )

    def set_char_color(self, col, env_ids):
        return
        # Not implemented yet

    def render(self):
        if not self.headless and self.init_done:
            if self.perspective_view is None:
                from phys_anim.envs.base_interface.isaaclab_utils.perspective_viewer import (
                    PerspectiveViewer,
                )

                self.perspective_view = PerspectiveViewer()
                self.init_camera()
            else:
                self.update_camera()
        super().render()

    def init_camera(self):
        self.cam_prev_char_pos = self.get_humanoid_root_states()[0, :3].cpu().numpy()
        pos = self.cam_prev_char_pos + np.array([0, -5, 1])
        self.perspective_view.set_camera_view(
            pos, self.cam_prev_char_pos + np.array([0, 0, 0.2])
        )

    def update_camera(self):
        char_root_pos = self.get_humanoid_root_states()[0, :3].cpu().numpy()
        cam_pos = np.array(self.perspective_view.get_camera_state())
        cam_delta = cam_pos - self.cam_prev_char_pos

        new_cam_target = np.array(
            [char_root_pos[0], char_root_pos[1], char_root_pos[2] + 0.2]
        )
        new_cam_pos = np.array(
            [
                char_root_pos[0] + cam_delta[0],
                char_root_pos[1] + cam_delta[1],
                char_root_pos[2] + cam_delta[2],
            ]
        )
        self.perspective_view.set_camera_view(new_cam_pos, new_cam_target)
        self.cam_prev_char_pos[:] = char_root_pos

    def output_motion(self):
        # TODO: add code to record states
        raise NotImplementedError
