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
"""Scene library for managing objects and scenes in environments.

This module provides the SceneLib class which manages object spawning, placement,
and interaction in simulation environments. It supports both static and dynamic objects,
motion-controlled objects, and complex multi-object scenes.


"""

import logging
import random
import copy
from dataclasses import dataclass, field, MISSING
from typing import List, Optional, Tuple, Dict, Union
from protomotions.utils import rotations
import torch
import numpy as np
import trimesh
import os
from enum import Enum
from protomotions.utils.config_builder import ConfigBuilder
from protomotions.utils.motion_interpolation_utils import calc_frame_blend
from protomotions.utils.mesh_utils import (
    as_mesh,
    compute_bounding_box,
)
from protomotions.simulator.base_simulator.simulator_state import (
    ObjectState,
    StateConversion,
)


from protomotions.components.terrains.terrain import Terrain

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ObjectOptions:
    """
    Contains options for configuring object properties in the simulator.
    """

    fix_base_link: bool = field(default=MISSING)
    vhacd_enabled: bool = None
    vhacd_params: Dict = field(
        default_factory=lambda: {
            "resolution": None,
            "max_convex_hulls": None,
            "max_num_vertices_per_ch": None,
        }
    )
    density: float = None
    angular_damping: float = None
    linear_damping: float = None
    max_angular_velocity: float = None
    texture_path: str = None  # Path to texture file

    def to_dict(self) -> Dict:
        """Convert options to a dictionary, excluding None values.

        This method recursively filters out None values from nested dictionaries
        to prevent type errors in simulator backends that don't accept None.
        """
        options_dict = {}
        for field_name, field_value in self.__dict__.items():
            if field_value is None:
                continue
            # Recursively filter None values from nested dictionaries
            if isinstance(field_value, dict):
                filtered_dict = {k: v for k, v in field_value.items() if v is not None}
                # Only include the dict if it has non-None values
                if filtered_dict:
                    options_dict[field_name] = filtered_dict
            else:
                options_dict[field_name] = field_value
        return options_dict


@dataclass
class SceneObject:
    """
    Represents an object inside a scene.

    The object can be static or have motion:
    - Static object: translation is a 3D vector and rotation is a 4D quaternion
    - Moving object: translation is a sequence of 3D vectors and rotation is a sequence of 4D quaternions

    Supported input formats (all converted to torch.Tensor internally):
    - Static: tuple(3), list(3), numpy.ndarray(3), torch.Tensor(3)
    - Motion: list[tuple(3)], numpy.ndarray(N,3), torch.Tensor(N,3)

    For moving objects, the first frame defines the initial state, and subsequent frames
    define the motion path. Both translation and rotation must have the same number of frames.

    fps: Frames per second for motion data. Must be defined if the object has motion.
         For static objects, fps defaults to 1.0.
    """

    translation: Union[
        Tuple[float, float, float],
        List[Tuple[float, float, float]],
        np.ndarray,
        torch.Tensor,
    ] = field(default=MISSING)
    rotation: Union[
        Tuple[float, float, float, float],
        List[Tuple[float, float, float, float]],
        np.ndarray,
        torch.Tensor,
    ] = field(default=MISSING)
    options: ObjectOptions = field(default_factory=ObjectOptions)
    fps: Optional[float] = None

    object_dims: Tuple[float, float, float, float, float, float] = (
        None  # min_x, max_x, min_y, max_y, min_z, max_z
    )
    object_pointcloud: torch.Tensor = None
    object_pointcloud_normals: torch.Tensor = (
        None  # Normals corresponding to each point in the pointcloud
    )
    is_first_instance: bool = (
        True  # Whether this is the first instance of this object type
    )
    first_instance_id: Optional[int] = (
        None  # ID of the first instance of this object type, some sims can ultilize this to make loading duplicate objects more efficient
    )
    instance_id: Optional[int] = None  # Unique ID for this object instance

    def __post_init__(self):
        """
        Validate data and convert translation and rotation to PyTorch tensors.
        Sets default values for fps based on whether the object has motion.
        """
        # Convert translation to tensor
        self.translation = self._convert_to_tensor(self.translation, expected_dim=3)

        # Convert rotation to tensor
        self.rotation = self._convert_to_tensor(self.rotation, expected_dim=4)

        # Validate shapes match
        if self.translation.shape[0] > 1:
            # This is motion data with multiple frames
            assert (
                self.translation.shape[0] == self.rotation.shape[0]
            ), f"Translation ({self.translation.shape[0]} frames) and rotation ({self.rotation.shape[0]} frames) must have the same number of frames"

            # For objects with motion, fps must be defined
            assert (
                self.fps is not None
            ), "FPS must be defined for objects with motion data"
        else:
            # Static object, set default fps
            self.fps = 1.0

    def _convert_to_tensor(self, data, expected_dim):
        """
        Convert input data to PyTorch tensor with appropriate shape.

        Args:
            data: Input data (tuple, list, numpy array, or torch tensor)
            expected_dim: Expected dimension of each vector (3 for translation, 4 for rotation)

        Returns:
            torch.Tensor: Data converted to tensor with proper shape
        """
        device = "cpu"  # Default device

        # Handle list of tuples/lists for motion data
        if torch.is_tensor(data):
            # If data is already a tensor, clone and detach to get an independent copy
            # Then ensure correct dtype and device, and reshape
            return (
                data.clone()
                .detach()
                .to(dtype=torch.float, device=device)
                .view(-1, expected_dim)
            )
        else:
            # If data is not a tensor (e.g., list, tuple, numpy array), use torch.tensor
            return torch.tensor(data, dtype=torch.float, device=device).view(
                -1, expected_dim
            )

    def has_motion(self) -> bool:
        """
        Return True if the object has motion data (multiple frames).
        """
        return self.translation.shape[0] > 1

    @property
    def start_pose(self) -> ObjectState:
        """
        Returns the initial pose (first frame) of the object as an ObjectState.

        For static objects, returns the single pose.
        For objects with motion, returns the first frame of motion.

        Returns:
            ObjectState: Object state containing translation and rotation tensors,
                        both with shape [1, 3] and [1, 4] respectively (batched format).
        """
        translation = self.translation[0]
        rotation = self.rotation[0]

        return ObjectState(
            root_pos=translation,
            root_rot=rotation,
            state_conversion=StateConversion.COMMON,
        )

    @property
    def object_identifier(self) -> str:
        """Generate a unique identifier for the object"""
        raise NotImplementedError("Subclasses must implement object_identifier")

    def calculate_dimensions(self) -> Tuple[float, float, float, float, float, float]:
        """Calculate the object dimensions - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement calculate_dimensions")

    def compute_pointcloud(self, pointcloud_samples_per_object: int):
        """Compute the pointcloud for the object"""
        raise NotImplementedError("Subclasses must implement compute_pointcloud")


@dataclass
class MeshSceneObject(SceneObject):
    """
    Represents a mesh object specified via a file path.
    """

    object_path: str = None

    def __post_init__(self):
        """Validate that object_path is specified, compute dimensions"""
        if self.object_path is None:
            raise ValueError("object_path must be specified for MeshSceneObject")

        # Calculate dimensions if not provided
        if self.object_dims is None:
            self.object_dims = self.calculate_dimensions()

        super().__post_init__()

    def calculate_dimensions(self) -> Tuple[float, float, float, float, float, float]:
        """Calculate the dimensions from the mesh file"""

        obj_path = (
            self.object_path.replace(".urdf", ".obj")
            .replace(".usda", ".obj")
            .replace(".usd", ".obj")
        )
        stl_path = (
            self.object_path.replace(".urdf", ".stl")
            .replace(".usda", ".stl")
            .replace(".usd", ".stl")
        )
        ply_path = (
            self.object_path.replace(".urdf", ".ply")
            .replace(".usda", ".ply")
            .replace(".usd", ".ply")
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
        else:
            raise FileNotFoundError(
                f"Object file not found: {obj_path} / {stl_path} / {ply_path}"
            )

        min_x = m_x
        max_x = min_x + w_x
        min_y = m_y
        max_y = min_y + w_y
        min_z = m_z
        max_z = min_z + w_z

        return min_x, max_x, min_y, max_y, min_z, max_z

    def compute_pointcloud(self, pointcloud_samples_per_object: int):
        """Compute the pointcloud for the object"""
        obj_path = (
            self.object_path.replace(".urdf", ".obj")
            .replace(".usda", ".obj")
            .replace(".usd", ".obj")
        )
        stl_path = (
            self.object_path.replace(".urdf", ".stl")
            .replace(".usda", ".stl")
            .replace(".usd", ".stl")
        )
        ply_path = (
            self.object_path.replace(".urdf", ".ply")
            .replace(".usda", ".ply")
            .replace(".usd", ".ply")
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
            # Sample points evenly from the mesh surface and get face indices
            point_cloud_np, face_indices = trimesh.sample.sample_surface_even(
                mesh, pointcloud_samples_per_object
            )

            if point_cloud_np.shape[0] < pointcloud_samples_per_object:
                # Even spacing uses rejection sampling, as a result it may return less points than requested
                # we add the extra points by randomly sampling the mesh surface again
                missing_points = pointcloud_samples_per_object - point_cloud_np.shape[0]
                extra_points_np, extra_face_indices = trimesh.sample.sample_surface(
                    mesh, missing_points
                )
                point_cloud_np = np.concatenate(
                    [point_cloud_np, extra_points_np], axis=0
                )
                face_indices = np.concatenate(
                    [face_indices, extra_face_indices], axis=0
                )

            # Get face normals corresponding to the sampled points
            face_normals = mesh.face_normals[face_indices]

        else:
            raise FileNotFoundError(
                f"Object file not found: {obj_path} / {stl_path} / {ply_path}"
            )

        self.object_pointcloud = torch.tensor(point_cloud_np, dtype=torch.float)
        self.object_pointcloud_normals = torch.tensor(face_normals, dtype=torch.float)

    @property
    def object_identifier(self) -> str:
        """Use object_path as the unique identifier for mesh objects"""
        return self.object_path


@dataclass
class PrimitiveSceneObject(SceneObject):
    """
    Base class for primitive shape objects.
    """

    def __post_init__(self):
        # Calculate dimensions if not provided
        if self.object_dims is None:
            self.object_dims = self.calculate_dimensions()

        super().__post_init__()

    @property
    def object_identifier(self) -> str:
        """Generate a unique identifier for primitive shapes based on type and dimensions"""
        # Each subclass should implement this method
        raise NotImplementedError("Subclasses must implement object_identifier")


@dataclass
class BoxSceneObject(PrimitiveSceneObject):
    """
    Represents a box primitive shape.
    """

    width: float = None
    depth: float = None
    height: float = None

    def __post_init__(self):
        """Validate dimensions"""
        if any(dim is None for dim in [self.width, self.depth, self.height]):
            raise ValueError("Box shape requires width, depth, and height dimensions")
        super().__post_init__()

    def calculate_dimensions(self) -> Tuple[float, float, float, float, float, float]:
        """Calculate box dimensions"""
        min_x, max_x = -self.width / 2, self.width / 2
        min_y, max_y = -self.depth / 2, self.depth / 2
        min_z, max_z = -self.height / 2, self.height / 2
        return (min_x, max_x, min_y, max_y, min_z, max_z)

    def compute_pointcloud(self, pointcloud_samples_per_object: int):
        """Compute pointcloud for box by sampling points on the surface."""

        # TODO: we probably should always include the corners, no matter what
        # Special case: if exactly 8 points requested, return the 8 corners
        if pointcloud_samples_per_object == 8:
            # Get the box dimensions
            # print("compute_pointcloud: 8 corner points of box requested")
            object_dims = self.calculate_dimensions()
            min_x, max_x, min_y, max_y, min_z, max_z = object_dims

            # Create the 8 corner points
            corners = torch.tensor(
                [
                    [min_x, min_y, min_z],  # 0: bottom, back, left
                    [max_x, min_y, min_z],  # 1: bottom, back, right
                    [min_x, max_y, min_z],  # 2: bottom, front, left
                    [max_x, max_y, min_z],  # 3: bottom, front, right
                    [min_x, min_y, max_z],  # 4: top, back, left
                    [max_x, min_y, max_z],  # 5: top, back, right
                    [min_x, max_y, max_z],  # 6: top, front, left
                    [max_x, max_y, max_z],  # 7: top, front, right
                ],
                dtype=torch.float,
            )

            # Normals for the 8 corners (average of adjacent faces) - approximated
            # For simplicity, assigning dominant face normal, though true corner normal is ambiguous
            corner_normals = torch.tensor(
                [
                    [-1, -1, -1],
                    [1, -1, -1],
                    [-1, 1, -1],
                    [1, 1, -1],  # Bottom 4
                    [-1, -1, 1],
                    [1, -1, 1],
                    [-1, 1, 1],
                    [1, 1, 1],  # Top 4
                ],
                dtype=torch.float,
            )
            corner_normals = corner_normals / torch.norm(
                corner_normals, dim=1, keepdim=True
            )  # Normalize

            self.object_pointcloud = corners
            self.object_pointcloud_normals = corner_normals
            return

        # Original implementation for when more than 8 points are requested
        object_dims = self.calculate_dimensions()

        # Calculate surface areas for each face pair
        width = object_dims[1] - object_dims[0]
        depth = object_dims[3] - object_dims[2]
        height = object_dims[5] - object_dims[4]

        # Calculate area of each face pair (front/back, left/right, top/bottom)
        area_x = 2 * (depth * height)  # front and back faces
        area_y = 2 * (width * height)  # left and right faces
        area_z = 2 * (width * depth)  # top and bottom faces
        total_area = area_x + area_y + area_z

        # Distribute points according to surface area
        points_x = int(pointcloud_samples_per_object * (area_x / total_area))
        points_y = int(pointcloud_samples_per_object * (area_y / total_area))
        points_z = (
            pointcloud_samples_per_object - points_x - points_y
        )  # Remaining points

        # Split points evenly between each pair of faces
        points_front = points_x // 2
        points_back = points_x - points_front
        points_right = points_y // 2
        points_left = points_y - points_right
        points_top = points_z // 2
        points_bottom = points_z - points_top

        faces_points = []
        faces_normals = []  # List to store normals for each face

        # Helper function to generate grid points on a face
        def generate_face_points(num_points, fixed_dim, fixed_val, var_dim1, var_dim2):
            # Calculate aspect ratio of the face
            width = var_dim1[1] - var_dim1[0]
            height = var_dim2[1] - var_dim2[0]
            aspect_ratio = width / height

            # Calculate points per side maintaining aspect ratio
            points_height = max(1, int(np.sqrt(num_points / aspect_ratio)))
            points_width = max(1, int(points_height * aspect_ratio))

            # Adjust if we're off from target number of points
            total_points = points_width * points_height
            if total_points < num_points:
                # Add extra points to the longer side
                if width > height:
                    points_width += 1
                else:
                    points_height += 1

            # Create perfectly even grid
            d1 = torch.linspace(var_dim1[0], var_dim1[1], points_width)
            d2 = torch.linspace(var_dim2[0], var_dim2[1], points_height)

            # Create meshgrid
            g1, g2 = torch.meshgrid(d1, d2, indexing="ij")

            # Create the points array based on which dimension is fixed
            if fixed_dim == 0:  # Fixed X
                points = torch.stack([torch.full_like(g1, fixed_val), g1, g2], dim=-1)
            elif fixed_dim == 1:  # Fixed Y
                points = torch.stack([g1, torch.full_like(g1, fixed_val), g2], dim=-1)
            else:  # Fixed Z
                points = torch.stack([g1, g2, torch.full_like(g1, fixed_val)], dim=-1)

            points = points.reshape(-1, 3)

            # If we have more points than needed, take evenly spaced points
            if points.shape[0] > num_points:
                step = points.shape[0] // num_points
                indices = torch.arange(0, points.shape[0], step)[:num_points]
                points = points[indices]

            return points

        # Generate points and normals for each face
        # Front face (fixed x = max_x), normal = (1, 0, 0)
        front_points = generate_face_points(
            points_front,
            0,
            object_dims[1],
            (object_dims[2], object_dims[3]),
            (object_dims[4], object_dims[5]),
        )
        faces_points.append(front_points)
        faces_normals.append(
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float).expand_as(front_points)
        )

        # Back face (fixed x = min_x), normal = (-1, 0, 0)
        back_points = generate_face_points(
            points_back,
            0,
            object_dims[0],
            (object_dims[2], object_dims[3]),
            (object_dims[4], object_dims[5]),
        )
        faces_points.append(back_points)
        faces_normals.append(
            torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float).expand_as(back_points)
        )

        # Right face (fixed y = max_y), normal = (0, 1, 0)
        right_points = generate_face_points(
            points_right,
            1,
            object_dims[3],
            (object_dims[0], object_dims[1]),
            (object_dims[4], object_dims[5]),
        )
        faces_points.append(right_points)
        faces_normals.append(
            torch.tensor([0.0, 1.0, 0.0], dtype=torch.float).expand_as(right_points)
        )

        # Left face (fixed y = min_y), normal = (0, -1, 0)
        left_points = generate_face_points(
            points_left,
            1,
            object_dims[2],
            (object_dims[0], object_dims[1]),
            (object_dims[4], object_dims[5]),
        )
        faces_points.append(left_points)
        faces_normals.append(
            torch.tensor([0.0, -1.0, 0.0], dtype=torch.float).expand_as(left_points)
        )

        # Top face (fixed z = max_z), normal = (0, 0, 1)
        top_points = generate_face_points(
            points_top,
            2,
            object_dims[5],
            (object_dims[0], object_dims[1]),
            (object_dims[2], object_dims[3]),
        )
        faces_points.append(top_points)
        faces_normals.append(
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float).expand_as(top_points)
        )

        # Bottom face (fixed z = min_z), normal = (0, 0, -1)
        bottom_points = generate_face_points(
            points_bottom,
            2,
            object_dims[4],
            (object_dims[0], object_dims[1]),
            (object_dims[2], object_dims[3]),
        )
        faces_points.append(bottom_points)
        faces_normals.append(
            torch.tensor([0.0, 0.0, -1.0], dtype=torch.float).expand_as(bottom_points)
        )

        # Combine all face points and normals
        point_cloud = torch.cat(faces_points, dim=0)
        point_cloud_normals = torch.cat(faces_normals, dim=0)

        # Final adjustment to match exactly the requested number of points
        if point_cloud.shape[0] > pointcloud_samples_per_object:
            # Take evenly spaced points
            step = point_cloud.shape[0] // pointcloud_samples_per_object
            indices = torch.arange(0, point_cloud.shape[0], step)[
                :pointcloud_samples_per_object
            ]
            point_cloud = point_cloud[indices]
            point_cloud_normals = point_cloud_normals[
                indices
            ]  # Adjust normals accordingly
        elif point_cloud.shape[0] < pointcloud_samples_per_object:
            # Duplicate points evenly from the existing points
            num_repeats = pointcloud_samples_per_object // point_cloud.shape[0]
            remainder = pointcloud_samples_per_object % point_cloud.shape[0]

            # First, repeat the entire point cloud
            point_cloud = point_cloud.repeat(num_repeats, 1)
            point_cloud_normals = point_cloud_normals.repeat(num_repeats, 1)

            # Then add the remaining points needed from the beginning
            if remainder > 0:
                extra_points = point_cloud[:remainder]
                extra_normals = point_cloud_normals[
                    :remainder
                ]  # Add corresponding normals
                point_cloud = torch.cat([point_cloud, extra_points], dim=0)
                point_cloud_normals = torch.cat(
                    [point_cloud_normals, extra_normals], dim=0
                )  # Add corresponding normals

        self.object_pointcloud = point_cloud
        self.object_pointcloud_normals = point_cloud_normals

    @property
    def object_identifier(self) -> str:
        """Generate a unique identifier for box shapes"""

        def format_float(val):
            if val is None:
                return "none"
            return str(val).replace(".", "_")

        return f"box_w{format_float(self.width)}_d{format_float(self.depth)}_h{format_float(self.height)}"


@dataclass
class SphereSceneObject(PrimitiveSceneObject):
    """
    Represents a sphere primitive shape.
    """

    radius: float = None

    def __post_init__(self):
        """Validate dimensions"""
        if self.radius is None:
            raise ValueError("Sphere shape requires a radius")
        super().__post_init__()

    def calculate_dimensions(self) -> Tuple[float, float, float, float, float, float]:
        """Calculate sphere dimensions"""
        r = self.radius
        return (-r, r, -r, r, -r, r)

    def compute_pointcloud(self, pointcloud_samples_per_object: int):
        """Compute pointcloud for sphere by sampling points on the surface."""
        # Generate points on unit sphere surface using fibonacci spiral
        golden_ratio = (1 + 5**0.5) / 2
        i = torch.arange(pointcloud_samples_per_object, dtype=torch.float)
        phi = 2 * torch.pi * (i / golden_ratio % 1)
        cos_theta = 1 - 2 * (i + 0.5) / pointcloud_samples_per_object
        theta = torch.arccos(cos_theta)

        # Convert spherical coordinates to cartesian
        x = torch.cos(phi) * torch.sin(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = cos_theta

        # Scale by radius and combine into point cloud
        point_cloud = torch.stack([x, y, z], dim=1) * self.radius
        self.object_pointcloud = point_cloud
        # Normals are the normalized position vectors
        self.object_pointcloud_normals = point_cloud / self.radius

    @property
    def object_identifier(self) -> str:
        """Generate a unique identifier for sphere shapes"""

        def format_float(val):
            if val is None:
                return "none"
            return str(val).replace(".", "_")

        return f"sphere_r{format_float(self.radius)}"


@dataclass
class CylinderSceneObject(PrimitiveSceneObject):
    """
    Represents a cylinder primitive shape.
    """

    radius: float = None
    height: float = None

    def __post_init__(self):
        """Validate dimensions"""
        if self.radius is None or self.height is None:
            raise ValueError("Cylinder shape requires radius and height dimensions")
        super().__post_init__()

    def calculate_dimensions(self) -> Tuple[float, float, float, float, float, float]:
        """Calculate cylinder dimensions"""
        r = self.radius
        h = self.height
        return (-r, r, -r, r, -h / 2, h / 2)

    def compute_pointcloud(self, pointcloud_samples_per_object: int):
        """Compute pointcloud for cylinder by sampling points on the surface."""
        # Determine number of points for each part (sides, top, bottom)
        side_points = int(pointcloud_samples_per_object * 0.6)  # 60% for the sides

        # Generate points for the cylindrical surface
        # Ensure we have at least 2 points for height and theta if side_points > 0
        num_height_points = max(2, int(np.sqrt(side_points))) if side_points > 0 else 0
        num_theta_points = (
            max(2, int(side_points / num_height_points)) if num_height_points > 0 else 0
        )
        actual_side_points = num_height_points * num_theta_points

        if actual_side_points > 0:
            height = torch.linspace(
                -self.height / 2, self.height / 2, num_height_points
            )
            theta = torch.linspace(0, 2 * torch.pi, num_theta_points + 1)[
                :-1
            ]  # Avoid duplicate end point
            h, t = torch.meshgrid(height, theta, indexing="ij")
            x_side = self.radius * torch.cos(t)
            y_side = self.radius * torch.sin(t)
            z_side = h
            side_points_tensor = torch.stack(
                [x_side.flatten(), y_side.flatten(), z_side.flatten()], dim=1
            )
            # Normals for side surface (x, y, 0) normalized
            side_normals = torch.stack(
                [
                    x_side.flatten(),
                    y_side.flatten(),
                    torch.zeros_like(z_side.flatten()),
                ],
                dim=1,
            )
            side_normals = side_normals / torch.norm(side_normals, dim=1, keepdim=True)
        else:
            side_points_tensor = torch.empty((0, 3), dtype=torch.float)
            side_normals = torch.empty((0, 3), dtype=torch.float)

        # Adjust cap points based on actual side points generated
        total_generated_points = actual_side_points
        remaining_points = pointcloud_samples_per_object - total_generated_points
        cap_points_each = max(0, remaining_points // 2)
        top_cap_points_count = cap_points_each
        bottom_cap_points_count = (
            pointcloud_samples_per_object
            - total_generated_points
            - top_cap_points_count
        )  # Ensure total count is correct

        # Generate points for top and bottom caps using spiral pattern
        def generate_cap_points(z_val, num_points):
            if num_points <= 0:
                return torch.empty((0, 3), dtype=torch.float)
            # Ensure sqrt calculation is safe
            num_sqrt = max(1, int(np.sqrt(num_points)))
            r = torch.sqrt(torch.linspace(0, 1, num_sqrt))
            theta = torch.linspace(
                0, 8 * torch.pi, num_sqrt
            )  # Use same number for theta for better grid
            r_grid, t_grid = torch.meshgrid(r, theta, indexing="ij")
            x = self.radius * r_grid * torch.cos(t_grid)
            y = self.radius * r_grid * torch.sin(t_grid)
            z = torch.full_like(x, z_val)
            cap_points_tensor = torch.stack(
                [x.flatten(), y.flatten(), z.flatten()], dim=1
            )
            # If we generated more points than needed due to grid, sample randomly
            if cap_points_tensor.shape[0] > num_points:
                indices = torch.randperm(cap_points_tensor.shape[0])[:num_points]
                cap_points_tensor = cap_points_tensor[indices]
            return cap_points_tensor

        top_points_tensor = generate_cap_points(self.height / 2, top_cap_points_count)
        bottom_points_tensor = generate_cap_points(
            -self.height / 2, bottom_cap_points_count
        )

        # Normals for caps
        top_normals = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float).expand_as(
            top_points_tensor
        )
        bottom_normals = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float).expand_as(
            bottom_points_tensor
        )

        # Combine all points and normals
        point_cloud = torch.cat(
            [side_points_tensor, top_points_tensor, bottom_points_tensor], dim=0
        )
        point_cloud_normals = torch.cat(
            [side_normals, top_normals, bottom_normals], dim=0
        )

        # Ensure the exact number of points if somehow undershot (unlikely with adjustments)
        current_points = point_cloud.shape[0]
        if current_points < pointcloud_samples_per_object:
            print(
                f"Warning: Cylinder point cloud generated {current_points} points, requested {pointcloud_samples_per_object}. Duplicating points."
            )
            needed = pointcloud_samples_per_object - current_points
            extra_indices = torch.randint(0, current_points, (needed,))
            point_cloud = torch.cat([point_cloud, point_cloud[extra_indices]], dim=0)
            point_cloud_normals = torch.cat(
                [point_cloud_normals, point_cloud_normals[extra_indices]], dim=0
            )

        self.object_pointcloud = point_cloud
        self.object_pointcloud_normals = point_cloud_normals

    @property
    def object_identifier(self) -> str:
        """Generate a unique identifier for cylinder shapes"""

        def format_float(val):
            if val is None:
                return "none"
            return str(val).replace(".", "_")

        return f"cylinder_r{format_float(self.radius)}_h{format_float(self.height)}"


@dataclass
class Scene:
    """
    Represents a scene consisting of one or more SceneObjects.
    An offset (x, y) indicates the scene's location.
    """

    objects: List[SceneObject] = field(default_factory=list)
    offset: Tuple[float, float] = (0.0, 0.0)

    humanoid_motion_id: int = (
        -1
    )  # specific human motion to use for this scene, -1 means None

    def add_object(self, scene_object: SceneObject):
        """Add an object to the scene."""
        self.objects.append(scene_object)


class ReplicationMethod(Enum):
    """Method for replicating scenes."""

    FIRST = "first"
    WEIGHTED = "weighted"
    RANDOM = "random"
    SEQUENTIAL = "sequential"

    @classmethod
    def from_str(cls, value: str) -> "ReplicationMethod":
        """Create enum from string, case-insensitive."""
        try:
            return next(
                member for member in cls if member.value.lower() == value.lower()
            )
        except StopIteration:
            raise ValueError(
                f"'{value}' is not a valid {cls.__name__}. "
                f"Valid values are: {[e.value for e in cls]}"
            )


class SubsetMethod(Enum):
    """Method for subsetting scenes."""

    FIRST = "first"
    RANDOM = "random"
    SEQUENTIAL = "sequential"

    @classmethod
    def from_str(cls, value: str) -> "SubsetMethod":
        """Create enum from string, case-insensitive."""
        try:
            return next(
                member for member in cls if member.value.lower() == value.lower()
            )
        except StopIteration:
            raise ValueError(
                f"'{value}' is not a valid {cls.__name__}. "
                f"Valid values are: {[e.value for e in cls]}"
            )


@dataclass
class SceneLibConfig(ConfigBuilder):
    """Configuration for scene library - static parameters only.

    Runtime parameters (num_envs, terrain, scene_weights) are passed to SceneLib constructor.

    The configuration includes options for loading scenes from files, controlling replication
    strategies, and setting pointcloud sampling parameters.
    """

    _target_: str = "protomotions.components.scene_lib.SceneLib"
    scene_file: str = None  # Renamed from 'file'
    subset_method: Union[SubsetMethod, List[int]] = SubsetMethod.FIRST
    replicate_method: ReplicationMethod = ReplicationMethod.WEIGHTED
    pointcloud_samples_per_object: Optional[int] = None
    num_objects_per_env: int = None


class SceneLib:
    """
    A scene library for managing object spawning, placement, and motion in simulation environments.

    SceneLib manages collections of scenes, each containing one or more objects. It handles:
    - Scene replication/subsetting across environments
    - Object motion data combination and interpolation
    - Pointcloud generation for object observations

    Usage Workflows:

    1. Programmatic scene creation (tutorials, testing)::

        config = SceneLibConfig(scene_file=None, replicate_method="random")
        scenes = [Scene(objects=[obj1, obj2]), ...]
        scene_lib = SceneLib(config, num_envs, scenes, device, terrain)

    2. Loading from file (production, training)::

        config = SceneLibConfig(scene_file="scenes.pt", replicate_method="weighted")
        scene_lib = SceneLib(config, num_envs, scenes=None, device, terrain, scene_weights)

    3. Saving scenes for reuse (static method - no SceneLib instance or config needed)::

        SceneLib.save_scenes_to_file(scenes, "scenes.pt")

        # Note: Saved files contain only scene data (objects, poses, motion).
        # SceneLibConfig parameters (replicate_method, subset_method, etc.) are NOT saved.
        # Provide a fresh config when loading to specify runtime behavior.


    Key Assumptions:
    - All scenes must have the same number of objects
    - Object order within scenes matters (used for indexing)
    - Scene offsets calculated from terrain layout if terrain provided
    - Static objects (fix_base_link=True) have single frame, dynamic can have motion
    - FPS required for objects with motion, defaults to 1.0 for static objects

    Internal State Tensors:

    Object Motion Data (combined from ORIGINAL scenes only, like MotionLib):
        _object_translations: torch.Tensor (total_frames, 3)
            All object translations from original scenes concatenated
        _object_rotations: torch.Tensor (total_frames, 4)
            All object rotations (quaternions xyzw) from original scenes concatenated
        _motion_lengths: torch.Tensor (num_original_objects,)
            Motion length in seconds for each original object
            num_original_objects = len(_original_scenes) * num_objects_per_scene
        _motion_starts: torch.Tensor (num_original_objects,)
            Starting frame index in concatenated arrays for each original object
        _motion_dts: torch.Tensor (num_original_objects,)
            Delta time (1/fps) per frame for each original object
        _motion_num_frames: torch.Tensor (num_original_objects,)
            Number of frames for each original object (1 for static, >1 for dynamic)

    Object Geometry Data (combined from ORIGINAL scenes only):
        _object_pointclouds: torch.Tensor (num_original_scenes, num_objects_per_scene, num_points, 3)
            Neutral (unposed) pointclouds for original objects in local coordinates
        _object_pointcloud_normals: torch.Tensor (num_original_scenes, num_objects_per_scene, num_points, 3)
            Surface normals for original objects

    Scene Organization:
        _scene_to_original_scene_id: torch.Tensor (num_envs,)
            Maps each replicated scene index to its original scene index
            Enables efficient indexing: replicated scenes share original scene data
        _is_static_object: torch.Tensor (num_original_scenes, objects_per_scene)
            Boolean mask indicating which objects in original scenes are static
        _scene_offsets: List[Tuple[float, float]]
            (x, y) offset for each replicated scene in world coordinates (num_envs entries)

    Static vs Dynamic Objects:
        - Static: options.fix_base_link == True, single frame, no motion
        - Dynamic: options.fix_base_link == False, can have multi-frame motion data
    """

    def __init__(
        self,
        config: "SceneLibConfig",
        num_envs: int = 0,
        scenes: Optional[List[Scene]] = None,
        device: str = "cpu",
        terrain: Optional[Terrain] = None,
        scene_weights: Optional[List[float]] = None,
    ):
        """Initialize SceneLib from config.

        Creates either a populated scene library (if config.scene_file is set or scenes provided) or
        an empty scene library (if config.scene_file is None and no scenes) following Null Object pattern.

        Args:
            config: SceneLibConfig (always required, scene_file can be None for empty)
            num_envs: Number of environments (runtime parameter).
            scenes: List of Scene objects for programmatic creation. Mutually exclusive with config.scene_file.
            device: Device for torch tensors (runtime parameter).
            terrain: Optional Terrain object. Can be None for data processing.
            scene_weights: Optional weights for weighted replication (runtime parameter, e.g., from checkpoints).
        """
        self.config = config
        self.device = device
        self.num_envs = num_envs
        self.terrain = terrain

        # Store original scenes (before replication/subsetting)
        self._original_scenes: List[Scene] = []
        # Processed scenes (after replication/subsetting)
        self.scenes: List[Scene] = []

        self.num_objects_per_scene = 0
        self._scene_offsets = []

        # Placeholders for aggregated motion data
        self._object_translations = None
        self._object_rotations = None
        self._object_pointclouds = None
        self._object_pointcloud_normals = None
        self._motion_lengths = None
        self._motion_starts = None
        self._motion_dts = None
        self._motion_num_frames = None

        # Handle empty scene library (Null Object pattern)
        if config.scene_file is None and scenes is None:
            print("Creating empty SceneLib (no scenes)")
            self._create_empty()
            return

        # Validate num_envs when creating non-empty scene library
        if num_envs <= 0:
            raise ValueError(
                f"num_envs must be > 0 when creating non-empty SceneLib, got {num_envs}"
            )

        # Load from file OR use provided scenes
        if config.scene_file is not None:
            if scenes is not None:
                raise ValueError(
                    "Cannot provide both config.scene_file and scenes parameter"
                )
            scenes = self._load_scenes_from_file(config.scene_file, device)
            logger.info(
                f"Loaded {len(scenes)} original scenes from {config.scene_file}"
            )

        # Validate scene weights match number of scenes
        if scene_weights is not None:
            if len(scene_weights) != len(scenes):
                raise ValueError(
                    f"Number of scene_weights ({len(scene_weights)}) must match "
                    f"number of original scenes ({len(scenes)})"
                )

        # Process pointclouds and instance tracking BEFORE deepcopy
        if self.config.pointcloud_samples_per_object is not None:
            for scene in scenes:
                for obj in scene.objects:
                    obj.compute_pointcloud(self.config.pointcloud_samples_per_object)

        # Process objects to set is_first_instance flags BEFORE deepcopy
        self._process_scene_objects_for_asset_tracking(scenes)

        # Store original scenes AFTER computing pointclouds and setting flags
        self._original_scenes = copy.deepcopy(scenes)

        self._create_scenes(scenes, scene_weights)

    def _create_empty(self):
        """Create an empty scene library with no scenes."""
        self._original_scenes = []
        self.scenes = []
        self.num_objects_per_scene = 0
        self._scene_offsets = []
        self._is_static_object = torch.empty(0, 0, dtype=torch.bool, device=self.device)
        self._scene_to_original_scene_id = torch.empty(
            0, dtype=torch.long, device=self.device
        )

        # Empty motion data tensors
        self._object_translations = torch.empty(0, 3, device=self.device)
        self._object_rotations = torch.empty(0, 4, device=self.device)
        self._motion_lengths = torch.empty(0, device=self.device)
        self._motion_starts = torch.empty(0, dtype=torch.long, device=self.device)
        self._motion_dts = torch.empty(0, device=self.device)
        self._motion_num_frames = torch.empty(0, dtype=torch.long, device=self.device)

        # Empty pointcloud data
        self._object_pointclouds = torch.empty(0, 0, 0, 3, device=self.device)
        self._object_pointcloud_normals = torch.empty(0, 0, 0, 3, device=self.device)

    @classmethod
    def empty(cls, num_envs: int, device: str, terrain=None):
        """Create an empty SceneLib with no scenes.

        Factory method for creating empty scene libraries in a concise way.
        Empty scenes don't need terrain, but can be provided if available.

        Args:
            num_envs: Number of environments
            device: PyTorch device
            terrain: Terrain instance (optional, not needed for empty SceneLib)

        Returns:
            Empty SceneLib instance
        """
        return cls(
            config=SceneLibConfig(scene_file=None),
            num_envs=num_envs,
            scenes=None,
            device=device,
            terrain=terrain,
        )

    @staticmethod
    def _load_scenes_from_file(file_path: str, device: str) -> List[Scene]:
        """Load original scenes from saved SceneLib file.

        Args:
            file_path: Path to the SceneLib file
            device: Device for loading (used for map_location)

        Returns:
            List[Scene]: Original scenes from the file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"SceneLib file not found: {file_path}")

        loaded_data = torch.load(file_path, map_location=device, weights_only=False)
        return SceneLib._deserialize_scenes_from_storage_static(
            loaded_data["original_scenes"]
        )

    def get_humanoid_motion_ids(self):
        humanoid_motion_ids = []
        for scene in self.scenes:
            humanoid_motion_ids.append(scene.humanoid_motion_id)

        print(f"humanoid_motion_ids: {humanoid_motion_ids}")

        if all(motion_id == -1 for motion_id in humanoid_motion_ids):
            return None
        elif all(motion_id >= 0 for motion_id in humanoid_motion_ids):
            return humanoid_motion_ids
        else:
            raise ValueError(
                "Humanoid motion ids must be either all -1 or all valid motion ids"
            )

    def _create_scenes(
        self, scenes: List[Scene], scene_weights: Optional[List[float]] = None
    ):
        """
        Create and assign scenes to environments using configurations from the Terrain object.

        If fewer scenes are provided than num_envs, replicate scenes using the given replicate_method.
        If more scenes are provided than num_envs, subset scenes using the given subset_method.

        Args:
            scenes: List of Scene objects to process and assign to environments.
            scene_weights: Optional weights for weighted replication. If None, uniform weights used.
        """

        # Clear previous scene data
        self.pointcloud_samples_per_object = self.config.pointcloud_samples_per_object
        self._scene_offsets = []

        assigned_scenes = scenes
        scene_to_original_ids = list(range(len(scenes)))  # Initially 1:1 mapping

        if len(assigned_scenes) > self.num_envs:
            assigned_scenes, scene_weights, scene_to_original_ids = self._subset_scenes(
                assigned_scenes,
                scene_weights,
                self.config.subset_method,
                scene_to_original_ids,
            )

        # Replicate scenes if needed and track original scene IDs
        if len(assigned_scenes) < self.num_envs:
            assigned_scenes, scene_to_original_ids = self._replicate_scenes(
                assigned_scenes,
                scene_weights,
                self.config.replicate_method,
                scene_to_original_ids,
            )

        # Assign scene offsets (with or without terrain validation)
        self._assign_scene_offsets(assigned_scenes, self.terrain)

        # Store mapping from replicated scene index to original scene index
        self._scene_to_original_scene_id = torch.tensor(
            scene_to_original_ids, dtype=torch.long, device=self.device
        )

        # Build static/dynamic tracking from ORIGINAL scenes for motion/pointcloud lookup
        self._build_static_object_mask()

        self.scenes = assigned_scenes

        # Combine from original scenes only (like MotionLib)
        if self.pointcloud_samples_per_object is not None:
            self.combine_object_pointclouds()
            self.combine_object_pointcloud_normals()
        self.combine_object_motions()

    def _process_scene_objects_for_asset_tracking(self, scenes: List[Scene]):
        """Process objects to set is_first_instance flags for asset loading.

        This sets flags on the provided scenes (should be called before deepcopy).
        The flags are used by simulators to determine which objects need asset loading.

        Args:
            scenes: Scenes to process (will be modified in-place)
        """
        first_instances = {}  # Map object identifiers to their first instance IDs
        instance_counter = 0

        for scene_idx, scene in enumerate(scenes):
            for obj_idx, obj in enumerate(scene.objects):
                object_identifier = obj.object_identifier

                # Set instance ID
                obj.instance_id = instance_counter
                instance_counter += 1

                if object_identifier not in first_instances:
                    # This is the first instance of this object type
                    obj.is_first_instance = True
                    obj.first_instance_id = obj.instance_id
                    first_instances[object_identifier] = obj.instance_id
                else:
                    # This is a subsequent instance of an existing object type
                    obj.is_first_instance = False
                    obj.first_instance_id = first_instances[object_identifier]

    def _build_static_object_mask(self):
        """Build mask for static objects from original scenes."""
        objects_per_scene = len(self._original_scenes[0].objects)
        num_original_scenes = len(self._original_scenes)

        self._is_static_object = torch.ones(
            (num_original_scenes, objects_per_scene),
            dtype=torch.bool,
            device=self.device,
        )

        for scene_idx, scene in enumerate(self._original_scenes):
            for obj_idx, obj in enumerate(scene.objects):
                self._is_static_object[scene_idx, obj_idx] = not obj.has_motion()

    def _replicate_scenes(
        self,
        scenes: List[Scene],
        scene_weights: List[float],
        replicate_method: ReplicationMethod,
        scene_to_original_ids: List[int],
    ) -> Tuple[List[Scene], List[int]]:
        """Replicate scenes if needed to match num_envs, tracking original scene IDs."""
        print(
            f"Replicating {self.num_envs - len(scenes)} scenes with method: {replicate_method}"
        )
        num_scenes = len(scenes)
        replicated_scenes = list(scenes)
        replicated_ids = list(scene_to_original_ids)

        if replicate_method == ReplicationMethod.SEQUENTIAL:
            for i in range(self.num_envs - num_scenes):
                idx = i % num_scenes
                scene = copy.deepcopy(replicated_scenes[idx])
                replicated_scenes.append(scene)
                replicated_ids.append(replicated_ids[idx])  # Track original ID
        elif replicate_method in [ReplicationMethod.RANDOM, ReplicationMethod.WEIGHTED]:
            if replicate_method == ReplicationMethod.RANDOM:
                scene_weights = None
            for _ in range(self.num_envs - num_scenes):
                idx = random.choices(range(num_scenes), weights=scene_weights, k=1)[0]
                scene = copy.deepcopy(replicated_scenes[idx])
                replicated_scenes.append(scene)
                replicated_ids.append(replicated_ids[idx])  # Track original ID
        else:
            logger.error("Unknown replicate method: %s", replicate_method)
            raise ValueError(
                "Replicate method must be either ReplicationMethod.SEQUENTIAL or ReplicationMethod.RANDOM."
            )

        return replicated_scenes, replicated_ids

    def _subset_scenes(
        self,
        scenes: List[Scene],
        scene_weights: List[float],
        subset_method: Union[SubsetMethod, List[int]],
        scene_to_original_ids: List[int],
    ):
        """Subset scenes if needed to match num_envs, tracking original scene IDs."""

        print(f"subset_method: {subset_method}")

        if subset_method == SubsetMethod.FIRST:
            return (
                scenes[: self.num_envs],
                scene_weights[: self.num_envs] if scene_weights is not None else None,
                scene_to_original_ids[: self.num_envs],
            )
        elif subset_method == SubsetMethod.LAST:
            return (
                scenes[-self.num_envs :],
                scene_weights[-self.num_envs :] if scene_weights is not None else None,
                scene_to_original_ids[-self.num_envs :],
            )
        elif subset_method == SubsetMethod.RANDOM:
            scene_indices = random.sample(range(len(scenes)), self.num_envs)
            return (
                [scenes[i] for i in scene_indices],
                [scene_weights[i] for i in scene_indices]
                if scene_weights is not None
                else None,
                [scene_to_original_ids[i] for i in scene_indices],
            )
        elif isinstance(subset_method, list):
            scene_indices = subset_method
            return (
                [scenes[i] for i in scene_indices],
                [scene_weights[i] for i in scene_indices]
                if scene_weights is not None
                else None,
                [scene_to_original_ids[i] for i in scene_indices],
            )
        else:
            logger.error("Unknown subset method: %s", subset_method)
            raise ValueError(
                "Subset method must be either SubsetMethod, or a list of scene indices."
            )

    def _assign_scene_offsets(
        self, scenes: List[Scene], terrain: Optional[Terrain] = None
    ):
        """
        Assign scene offsets to environments, with optional terrain validation.

        Args:
            scenes: List of Scene objects to assign offsets for.
            terrain: Optional Terrain object for validation and offset calculation.
                     If None, uses pre-set scene offsets or defaults to (0.0, 0.0).
        """
        for idx, scene in enumerate(scenes):
            if terrain is not None:
                # Calculate offsets based on terrain layout
                x_offset = (
                    (idx % terrain.num_scenes_per_column + 1)
                    * terrain.spacing_between_scenes
                    + terrain.border * terrain.horizontal_scale
                )
                y_offset = (
                    idx // terrain.num_scenes_per_column + 1
                ) * terrain.spacing_between_scenes + terrain.scene_y_offset
                scene.offset = (x_offset, y_offset)

                # Validate with terrain
                scene_x = int(x_offset / terrain.horizontal_scale)
                scene_y = int(y_offset / terrain.horizontal_scale)
                locations = torch.tensor([[scene_x, scene_y]], device=terrain.device)
                assert (
                    terrain.is_valid_spawn_location(locations).cpu().item()
                ), f"Scene {idx} is not a valid spawn location."
                terrain.mark_scene_location(scene_x, scene_y)
            else:
                # No terrain - use pre-set offset or default to (0, 0)
                if not hasattr(scene, "offset") or scene.offset is None:
                    scene.offset = (0.0, 0.0)

            self._scene_offsets.append(scene.offset)

        # Ensure each scene has the same number of objects
        object_counts = [len(scene.objects) for scene in scenes]
        if len(set(object_counts)) != 1:
            logger.error(
                "All scenes must have the same number of objects. Found counts: %s",
                object_counts,
            )
            raise ValueError(
                "Scenes have inconsistent number of objects: " + str(object_counts)
            )
        self.num_objects_per_scene = object_counts[0]

    def combine_object_pointclouds(self):
        """Combine pointclouds from ORIGINAL scenes only (not replicated)."""
        all_pointclouds = []
        for scene in self._original_scenes:
            for obj in scene.objects:
                all_pointclouds.append(obj.object_pointcloud)
        num_original_scenes = len(self._original_scenes)
        self._object_pointclouds = (
            torch.cat(all_pointclouds, dim=0)
            .reshape(num_original_scenes, self.num_objects_per_scene, -1, 3)
            .to(self.device)
        )

    def combine_object_pointcloud_normals(self):
        """Combine pointcloud normals from ORIGINAL scenes only (not replicated)."""
        all_normals = []
        for scene in self._original_scenes:
            for obj in scene.objects:
                if obj.object_pointcloud_normals is None:
                    raise ValueError(
                        f"Object {obj.instance_id} ({obj.object_identifier}) is missing pointcloud normals."
                    )
                all_normals.append(obj.object_pointcloud_normals)

        num_original_scenes = len(self._original_scenes)
        self._object_pointcloud_normals = (
            torch.stack(
                all_normals, dim=0
            )  # Use stack instead of cat if shapes are guaranteed [N_points, 3]
            .reshape(
                num_original_scenes, self.num_objects_per_scene, -1, 3
            )  # Reshape assuming N_points is constant
            .to(self.device)
        )

    def combine_object_motions(self):
        """
        Combine motion data from ORIGINAL SceneObjects into unified tensors.

        Like MotionLib, this combines data from original scenes only (not replicated).
        Replicated scenes map to original object indices via _scene_to_original_scene_id.

        For each SceneObject in original scenes:
          - If motion is provided (multiple frames), all frames are processed.
          - Otherwise, a default static frame (using the object's translation and rotation) is added.

        The following tensors are created and stored in SceneLib:
            - self._object_translations: (total_frames, 3) - From original scenes only
            - self._object_rotations: (total_frames, 4) - From original scenes only
            - self._motion_lengths: (num_original_objects,) - Length per original object
            - self._motion_starts: (num_original_objects,) - Starting indices
            - self._motion_dts: (num_original_objects,) - Delta time per object
            - self._motion_num_frames: (num_original_objects,) - Frames per object
        """
        all_translations = []  # List of tensors to concatenate
        all_rotations = []  # List of tensors to concatenate
        motion_lengths_list = []
        motion_dts_list = []
        motion_num_frames_list = []
        motion_starts = {}  # Map object index -> starting index
        current_start = 0

        # Combine from ORIGINAL scenes only (like MotionLib)
        all_objects = []
        for scene in self._original_scenes:
            for obj in scene.objects:
                all_objects.append(obj)

        for idx, obj in enumerate(all_objects):
            motion_starts[idx] = current_start

            fps = obj.fps
            dt = 1.0 / fps

            # Get number of frames
            num_frames = obj.translation.shape[0]
            motion_length = num_frames * dt
            motion_lengths_list.append(motion_length)
            motion_dts_list.append(dt)
            motion_num_frames_list.append(num_frames)

            # Add all frames at once to our list of tensors
            all_translations.append(obj.translation.to(device=self.device))
            all_rotations.append(obj.rotation.to(device=self.device))

            current_start += num_frames

        # Concatenate all the tensors at once
        if all_translations:
            self._object_translations = torch.cat(all_translations, dim=0)
            self._object_rotations = torch.cat(all_rotations, dim=0)
        else:
            self._object_translations = torch.empty((0, 3), device=self.device)
            self._object_rotations = torch.empty((0, 4), device=self.device)

        self._motion_lengths = torch.tensor(
            motion_lengths_list, dtype=torch.float, device=self.device
        )
        num_objects = len(all_objects)
        self._motion_starts = torch.tensor(
            [motion_starts[i] for i in range(num_objects)],
            dtype=torch.long,
            device=self.device,
        )
        self._motion_dts = torch.tensor(
            motion_dts_list, dtype=torch.float, device=self.device
        )
        self._motion_num_frames = torch.tensor(
            motion_num_frames_list, dtype=torch.long, device=self.device
        )

        logger.info(
            "Combined motion data for %d objects with total frames: %d",
            num_objects,
            self._object_translations.shape[0],
        )

    def _calc_frame_blend(
        self,
        time: torch.Tensor,
        length: torch.Tensor,
        num_frames: torch.Tensor,
        dt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate frame indices and blend factor for interpolation.

        Args:
            time (torch.Tensor): Current time.
            length (torch.Tensor): Length of the motion sequence in seconds.
            num_frames (torch.Tensor): Number of frames in the motion sequence.
            dt (torch.Tensor): Time step between frames.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Frame index 0, frame index 1, and blend factor.
        """
        phase = time / length
        phase = torch.clip(phase, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt

        return frame_idx0, frame_idx1, blend

    def get_object_pose(
        self, object_indices: torch.Tensor, time: torch.Tensor
    ) -> ObjectState:
        """
        Get the interpolated poses for multiple objects at given times.

        Args:
            object_indices (torch.Tensor): 1D tensor of object indices with length equal to num_envs.
            time (torch.Tensor): 1D tensor of times at which to interpolate poses, length equal to num_envs.

        Returns:
            ObjectState: A batched ObjectState with translations (num_envs, 3) and rotations (num_envs, 4).
        """
        if self._motion_starts is None:
            logger.error(
                "Motion data not combined. Call combine_object_motions() first."
            )
            raise ValueError("Motion data not combined.")

        dt = self._motion_dts[object_indices]
        length = self._motion_lengths[object_indices]
        num_frames = self._motion_num_frames[object_indices]
        start_idx = self._motion_starts[object_indices]

        t_tensor = time.to(dtype=torch.float, device=self.device)
        frame_idx0, frame_idx1, blend = calc_frame_blend(
            t_tensor, length, num_frames, dt
        )
        idx0 = start_idx + frame_idx0
        idx1 = start_idx + frame_idx1

        translation0 = self._object_translations[idx0]
        translation1 = self._object_translations[idx1]
        translation = (1 - blend.unsqueeze(-1)) * translation0 + blend.unsqueeze(
            -1
        ) * translation1

        rotation0 = self._object_rotations[idx0]
        rotation1 = self._object_rotations[idx1]

        # Apply slerp to all rotations at once using proper tensor shapes
        # Add batch dimension to blend for broadcasting
        blend_for_slerp = blend.unsqueeze(-1).unsqueeze(-1)  # Shape: [batch, 1, 1]

        rotation = rotations.slerp(
            rotation0.unsqueeze(1), rotation1.unsqueeze(1), blend_for_slerp
        )
        rotation = rotation.squeeze(1)  # Remove the added dimension

        return ObjectState(
            root_pos=translation,
            root_rot=rotation,
            state_conversion=StateConversion.COMMON,
        )

    def get_scene_pose(
        self,
        scene_indices: torch.Tensor,
        time: torch.Tensor,
        respawn_offset: float = 0.0,
    ) -> ObjectState:
        """
        Get the interpolated poses for all objects in the specified scenes at given times.

        Args:
            scene_indices (torch.Tensor): 1D tensor of scene indices (can be replicated scenes).
            time (torch.Tensor): 1D tensor of times at which to interpolate poses, should match length of scene_indices.
            respawn_offset (float): Z-offset to apply to non-static objects.

        Returns:
            ObjectState: An ObjectState with tensors of shape:
                - translations: [num_scenes, objects_per_scene, 3]
                - rotations: [num_scenes, objects_per_scene, 4]
        """
        # Handle empty scene library
        if self.num_scenes() == 0:
            num_scenes = scene_indices.shape[0] if scene_indices.numel() > 0 else 0
            return ObjectState(
                root_pos=torch.zeros(num_scenes, 0, 3, device=self.device),
                root_rot=torch.zeros(num_scenes, 0, 4, device=self.device),
                root_vel=torch.zeros(num_scenes, 0, 3, device=self.device),
                root_ang_vel=torch.zeros(num_scenes, 0, 3, device=self.device),
                state_conversion=StateConversion.COMMON,
            )

        if self._motion_starts is None:
            logger.error(
                "Motion data not combined. Call combine_object_motions() first."
            )
            raise ValueError("Motion data not combined.")

        num_scenes = scene_indices.shape[0]
        objects_per_scene = self.num_objects_per_scene

        # Map scene indices to original scene indices
        original_scene_indices = self._scene_to_original_scene_id[scene_indices]

        # Calculate object indices in the original (combined) data
        # Shape: [num_scenes, objects_per_scene]
        batch_object_indices = original_scene_indices.unsqueeze(
            1
        ) * self.num_objects_per_scene + torch.arange(
            self.num_objects_per_scene, device=self.device
        ).unsqueeze(0)

        # Reshape to a flat tensor - shape: [num_scenes * objects_per_scene]
        batch_object_indices = batch_object_indices.reshape(-1)

        # Repeat each time value for each object in the scene - shape: [num_scenes * objects_per_scene]
        batch_times = time.repeat_interleave(objects_per_scene)

        # Get poses for all objects in one batch operation
        batch_poses = self.get_object_pose(batch_object_indices, batch_times)

        # Reshape the results to [num_scenes, objects_per_scene, 3] and [num_scenes, objects_per_scene, 4]
        all_translations = batch_poses.root_pos.reshape(
            num_scenes, objects_per_scene, 3
        )
        # Get static mask from original scenes
        static_mask = self._is_static_object[original_scene_indices]

        # Create an offset matrix and apply it only to non-static objects
        offset_matrix = torch.zeros_like(all_translations)
        offset_matrix[..., 2] = respawn_offset  # Set z-coordinate to respawn_offset

        # Apply the mask to only affect non-static objects (expand mask to match dimensions)
        mask = (~static_mask).unsqueeze(-1).expand_as(offset_matrix)
        offset_matrix = offset_matrix * mask

        # Add the offset
        all_translations = all_translations + offset_matrix

        all_rotations = batch_poses.root_rot.reshape(num_scenes, objects_per_scene, 4)

        return ObjectState(
            root_pos=all_translations,
            root_rot=all_rotations,
            state_conversion=StateConversion.COMMON,
        )

    def get_scene_neutral_pointcloud(
        self, scene_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get the neutral pointcloud for all objects in the specified scenes.

        Args:
            scene_indices: Scene indices (can be replicated). If None, returns all original scenes.

        Returns:
            torch.Tensor: Pointclouds for the specified scenes
        """
        if not hasattr(self, "_object_pointclouds") or self._object_pointclouds is None:
            logger.error(
                "object_pointclouds not initialized. Make sure combine_object_pointclouds was called."
            )
            raise ValueError("Scene object pointclouds not initialized.")

        if scene_indices is None:
            return self._object_pointclouds

        # Map to original scene indices
        original_scene_indices = self._scene_to_original_scene_id[scene_indices]
        return self._object_pointclouds[original_scene_indices]

    def get_scene_neutral_pointcloud_normals(
        self, scene_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get the neutral pointcloud normals for all objects in the specified scenes.

        Args:
            scene_indices: Scene indices (can be replicated). If None, returns all original scenes.

        Returns:
            torch.Tensor: Pointcloud normals for the specified scenes
        """
        if (
            not hasattr(self, "_object_pointcloud_normals")
            or self._object_pointcloud_normals is None
        ):
            logger.error(
                "object_pointcloud_normals not initialized. Make sure create_scenes was called and pointcloud generation was enabled."
            )
            raise ValueError("Scene object pointcloud normals not initialized.")

        if scene_indices is None:
            return self._object_pointcloud_normals

        # Map to original scene indices
        original_scene_indices = self._scene_to_original_scene_id[scene_indices]
        return self._object_pointcloud_normals[original_scene_indices]

    def num_scenes(self) -> int:
        """
        Returns the number of scenes in the library.

        This is the number of actual Scene objects AFTER replication
        Also equals num_envs with current system assumption (one scene assigned per environment)
        But originally there might be only 5 unique scenes that get replicated to 1000 envs,
        """

        return len(self.scenes)

    @property
    def scene_offsets(self) -> List[Tuple[float, float]]:
        """Returns the list of scene offsets."""
        return self._scene_offsets

    def get_scene_positions(
        self, terrain: Terrain, device: Optional[str] = None
    ) -> torch.Tensor:
        """Calculate scene positions in the object playground.

        Scenes are always placed in the object playground region of the terrain (ref terrain.py),
        which is always flat at z=0. This returns the [x, y, 0.0] position for each scene.

        Note: The object playground is a dedicated flat region appended to the terrain grid
        specifically for scene placement. It is not affected by terrain generation and
        always remains at ground level (z=0).

        Args:
            terrain: Terrain object (not used for height since playground is always flat)
            device: Device for tensor creation. If None, uses self.device

        Returns:
            torch.Tensor: Scene positions with shape (num_envs, 3) containing [x, y, 0.0]
        """
        if device is None:
            device = self.device

        # Handle empty scene library
        if self.num_scenes() == 0:
            return torch.zeros(self.num_envs, 3, device=device, dtype=torch.float)

        # Stack all scene offsets (x, y) and append z=0 (object playground is always flat)
        scene_xy = torch.tensor(self._scene_offsets, device=device, dtype=torch.float)
        scene_z = torch.zeros(self.num_envs, 1, device=device, dtype=torch.float)
        return torch.cat([scene_xy, scene_z], dim=1)

    # def get_object_dims_tensor(self, device: Optional[str] = None) -> torch.Tensor:
    #     """Get object dimensions as a batched tensor for all environments.

    #     Returns dimensions for all objects mapped through scene replication.
    #     Each object has 6 dimensions: [min_x, max_x, min_y, max_y, min_z, max_z]

    #     Args:
    #         device: Device for tensor creation. If None, uses self.device

    #     Returns:
    #         torch.Tensor: Object dimensions with shape (num_envs, num_objects_per_scene, 6)
    #     """
    #     if device is None:
    #         device = self.device

    #     # Get dims from original scenes
    #     original_dims = []
    #     for scene in self._original_scenes:
    #         scene_dims = []
    #         for obj in scene.objects:
    #             dims = torch.tensor(obj.object_dims, dtype=torch.float, device=device)
    #             scene_dims.append(dims)
    #         original_dims.append(torch.stack(scene_dims))

    #     original_dims = torch.stack(original_dims)  # (num_original_scenes, objects_per_scene, 6)

    #     # Map to replicated scenes
    #     return original_dims[self._scene_to_original_scene_id]  # (num_envs, objects_per_scene, 6)

    def get_default_object_state(self, device: Optional[str] = None) -> ObjectState:
        """Get default object state for environment resets.

        Computes initial poses using get_scene_pose() at time=0

        For empty SceneLib (no scenes), returns ObjectState with empty tensors.

        Args:
            device: Device for tensor creation. If None, uses self.device

        Returns:
            ObjectState: Default object state with:
                - root_pos: (num_envs, num_objects_per_scene, 3) - Global positions
                - root_rot: (num_envs, num_objects_per_scene, 4) - Quaternions
                - root_vel: (num_envs, num_objects_per_scene, 3) - Zeros
                - root_ang_vel: (num_envs, num_objects_per_scene, 3) - Zeros
        """
        if device is None:
            device = self.device

        # Get all scene indices
        all_scene_indices = torch.arange(self.num_envs, device=device)

        # Get poses at time=0 (initial poses from motion data)
        # This handles empty scenes internally
        object_states = self.get_scene_pose(
            scene_indices=all_scene_indices,
            time=torch.zeros(self.num_envs, device=device),
            respawn_offset=0.0,
        )

        # Initialize velocities to zero (already correct shape from get_scene_pose)
        object_states.root_vel = torch.zeros_like(object_states.root_pos)
        object_states.root_ang_vel = torch.zeros_like(object_states.root_pos)

        return object_states

    @staticmethod
    def save_scenes_to_file(scenes: List[Scene], file_path: str):
        """Save scenes to file without creating a SceneLib instance.

        This is a convenience static method for saving scenes that were created
        programmatically. No SceneLib instance or config is needed - just pass
        the scenes directly.

        The saved file contains only scene data (objects, poses, motion) and is
        independent of any SceneLibConfig parameters. When loading, you provide
        a fresh config with desired runtime parameters (num_envs, replicate_method, etc.).

        Args:
            scenes: List of Scene objects to save
            file_path: Path to save the scenes file (.pt)

        Raises:
            ValueError: If scenes have inconsistent number of objects
            AssertionError: If file_path doesn't end with .pt

        Example:
            ```python
            scenes = [Scene(objects=[obj1, obj2]), ...]
            SceneLib.save_scenes_to_file(scenes, "my_scenes.pt")
            ```
        """
        assert file_path.endswith(".pt"), "File path must end with .pt"

        # Validate all scenes have same number of objects
        object_counts = [len(scene.objects) for scene in scenes]
        if len(set(object_counts)) != 1:
            raise ValueError(
                f"All scenes must have the same number of objects. Found counts: {set(object_counts)}"
            )

        save_data = {
            "original_scenes": SceneLib._serialize_scenes_for_storage_static(scenes),
            "num_original_scenes": len(scenes),
            "num_objects_per_scene": object_counts[0],
        }

        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        torch.save(save_data, file_path)
        logger.info(f"Saved {len(scenes)} scenes to {file_path}")

    @staticmethod
    def _serialize_scenes_for_storage_static(
        scenes_to_serialize: List[Scene],
    ) -> List[Dict]:
        """Serialize scenes to a format suitable for storage (static method).

        Args:
            scenes_to_serialize: List of scenes to serialize.
        """

        serialized_scenes = []
        for scene in scenes_to_serialize:
            scene_data = {
                "offset": scene.offset,
                "humanoid_motion_id": scene.humanoid_motion_id,
                "objects": [],
            }
            for obj in scene.objects:
                obj_data = {
                    "type": obj.__class__.__name__,
                    "translation": obj.translation.cpu().numpy().tolist(),
                    "rotation": obj.rotation.cpu().numpy().tolist(),
                    "fps": obj.fps,
                    "object_dims": obj.object_dims,
                    "options": {
                        field_name: getattr(obj.options, field_name)
                        for field_name in obj.options.__dict__
                        if getattr(obj.options, field_name) is not None
                    },
                }
                # Type-specific properties
                if isinstance(obj, BoxSceneObject):
                    obj_data.update(
                        {"width": obj.width, "depth": obj.depth, "height": obj.height}
                    )
                elif isinstance(obj, SphereSceneObject):
                    obj_data.update({"radius": obj.radius})
                elif isinstance(obj, CylinderSceneObject):
                    obj_data.update({"radius": obj.radius, "height": obj.height})
                elif isinstance(obj, MeshSceneObject):
                    obj_data.update({"object_path": obj.object_path})
                scene_data["objects"].append(obj_data)
            serialized_scenes.append(scene_data)
        return serialized_scenes

    @staticmethod
    def _deserialize_scenes_from_storage_static(
        serialized_scenes: List[Dict],
    ) -> List[Scene]:
        """Static method to deserialize scenes from storage format."""
        scenes = []
        for scene_data in serialized_scenes:
            objects = []
            for obj_data in scene_data["objects"]:
                options = ObjectOptions(**obj_data["options"])
                obj_type = obj_data["type"]

                translation = torch.tensor(obj_data["translation"], dtype=torch.float)
                rotation = torch.tensor(obj_data["rotation"], dtype=torch.float)

                if obj_type == "BoxSceneObject":
                    obj = BoxSceneObject(
                        width=obj_data["width"],
                        depth=obj_data["depth"],
                        height=obj_data["height"],
                        translation=translation,
                        rotation=rotation,
                        fps=obj_data["fps"],
                        options=options,
                    )
                elif obj_type == "SphereSceneObject":
                    obj = SphereSceneObject(
                        radius=obj_data["radius"],
                        translation=translation,
                        rotation=rotation,
                        fps=obj_data["fps"],
                        options=options,
                    )
                elif obj_type == "CylinderSceneObject":
                    obj = CylinderSceneObject(
                        radius=obj_data["radius"],
                        height=obj_data["height"],
                        translation=translation,
                        rotation=rotation,
                        fps=obj_data["fps"],
                        options=options,
                    )
                elif obj_type == "MeshSceneObject":
                    obj = MeshSceneObject(
                        object_path=obj_data["object_path"],
                        translation=translation,
                        rotation=rotation,
                        fps=obj_data["fps"],
                        options=options,
                    )

                objects.append(obj)

            scene = Scene(
                objects=objects,
                offset=scene_data["offset"],
                humanoid_motion_id=scene_data["humanoid_motion_id"],
            )
            scenes.append(scene)

        return scenes


# ----------------------------------------------------------------------------
# Example usage:
if __name__ == "__main__":
    import torch

    # Define a dummy Terrain for example usage
    class DummyTerrain:
        def __init__(self):
            self.num_scenes_per_column = 2
            self.spacing_between_scenes = 5.0
            self.border = 2.0
            self.horizontal_scale = 1.0
            self.scene_y_offset = 0.0
            self.device = "cpu"

        def is_valid_spawn_location(self, locations):
            return torch.tensor(True)

        def mark_scene_location(self, x, y):
            pass

    terrain = DummyTerrain()

    # Create SceneObjects with options
    obj1 = MeshSceneObject(
        object_path="examples/data/armchair.obj",
        translation=(1.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        # fps will default to 1.0 for static object
        options=ObjectOptions(
            vhacd_enabled=True,
            vhacd_params={
                "resolution": 50000,
                "max_convex_hulls": 128,
                "max_num_vertices_per_ch": 64,
            },
            fix_base_link=True,
        ),
    )

    # After initialization, obj1.translation and obj1.rotation are PyTorch tensors
    print(
        f"obj1.translation is now a {type(obj1.translation)} with shape {obj1.translation.shape}"
    )
    print(
        f"obj1.rotation is now a {type(obj1.rotation)} with shape {obj1.rotation.shape}"
    )

    # Example with motion data (using MeshSceneObject)
    obj2 = MeshSceneObject(
        object_path="examples/data/armchair.urdf",
        translation=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        rotation=np.array(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
        ),
        fps=30.0,  # Required for motion
        options=ObjectOptions(
            vhacd_enabled=True, vhacd_params={"resolution": 50000}, fix_base_link=True
        ),
    )

    # Create a primitive box
    obj3 = BoxSceneObject(
        width=1.0,
        depth=1.0,
        height=1.0,
        translation=(3.0, 0.0, 0.5),
        rotation=(0.0, 0.0, 0.0, 1.0),
        options=ObjectOptions(fix_base_link=True),
    )

    scene1 = Scene(objects=[obj1, obj3])

    obj4 = MeshSceneObject(
        object_path="examples/data/elephant.stl",
        translation=(2.0, 2.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        options=ObjectOptions(
            vhacd_enabled=True, vhacd_params={"resolution": 50000}, fix_base_link=True
        ),
    )
    obj5 = MeshSceneObject(
        object_path="examples/data/armchair.obj",
        translation=(2.5, 2.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        options=ObjectOptions(
            vhacd_enabled=True, vhacd_params={"resolution": 50000}, fix_base_link=True
        ),
    )
    scene2 = Scene(objects=[obj4, obj5])

    scenes = [scene1, scene2]

    terrain = DummyTerrain()

    # Create config
    scene_lib_config = SceneLibConfig(
        scene_file=None,
        replicate_method=ReplicationMethod.RANDOM,
        subset_method=SubsetMethod.FIRST,
        pointcloud_samples_per_object=None,
    )

    # Create SceneLib
    scene_lib = SceneLib(
        config=scene_lib_config,
        num_envs=4,
        scenes=scenes,
        device="cpu",
        terrain=terrain,
    )

    for idx, scene in enumerate(scene_lib.scenes):
        logger.info(
            "Environment %d assigned Scene with objects %s with offset %s",
            idx,
            scene.objects,
            scene.offset,
        )

    # get_object_pose returns an ObjectState
    time = 1.0 / 30 * 0.5
    pose_obj0 = scene_lib.get_object_pose(
        object_indices=torch.tensor([0]), time=torch.tensor([time])
    )
    logger.info(
        "Pose for object at index 0 at time %s:\nTranslations: %s\nRotations: %s",
        time,
        pose_obj0.root_pos,
        pose_obj0.root_rot,
    )
