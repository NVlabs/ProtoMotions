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
import torch
from torch import Tensor
import numpy as np
from protomotions.utils import rotations
from scipy import ndimage


@torch.jit.script
def get_heights_jit(
    locations: Tensor,
    height_samples: Tensor,
    horizontal_scale: float,
):
    num_envs = locations.shape[0]
    if len(locations.shape) == 2:
        locations = locations.unsqueeze(1)
    num_samples_per_env = locations.shape[1]

    points = locations[..., :2].clone().reshape(num_envs, num_samples_per_env, 2)
    points = points / horizontal_scale
    floored_points = points.long()
    # this encompasses 4 possible points.
    # points are the top left corner of the 4 points
    # we will interpolate between the 4 points to get the height
    px = floored_points[:, :, 0].view(-1)
    py = floored_points[:, :, 1].view(-1)
    px = torch.clip(px, 0, height_samples.shape[0] - 2)
    py = torch.clip(py, 0, height_samples.shape[1] - 2)

    # Calculate the fractional part of the points' positions
    fx = points[:, :, 0].view(-1) - px.float()
    fy = points[:, :, 1].view(-1) - py.float()

    # Get the heights of the four surrounding points
    h_tl = height_samples[px, py]  # Top-left
    h_tr = height_samples[px + 1, py]  # Top-right
    h_bl = height_samples[px, py + 1]  # Bottom-left
    h_br = height_samples[px + 1, py + 1]  # Bottom-right

    # Perform bilinear interpolation
    h_t = h_tl + (h_tr - h_tl) * fx  # Top interpolation
    h_b = h_bl + (h_br - h_bl) * fx  # Bottom interpolation
    interpolated_heights = h_t + (h_b - h_t) * fy  # Final interpolation

    return interpolated_heights.view(num_envs, -1)


@torch.jit.script_if_tracing
def get_height_maps_jit(
    base_rot: Tensor,
    base_pos: Tensor,
    height_points: Tensor,
    height_samples: Tensor,
    num_height_points: int,
    terrain_horizontal_scale: float,
    w_last: bool,
    return_all_dims: bool,
):
    num_envs = base_rot.shape[0]

    points = rotations.quat_apply_yaw(
        base_rot.repeat(1, num_height_points), height_points, w_last
    ) + (base_pos[:, :3]).unsqueeze(1)

    points = points / terrain_horizontal_scale
    floored_points = points.long()
    # this encompasses 4 possible points.
    # points are the top left corner of the 4 points
    # we will interpolate between the 4 points to get the height
    px = floored_points[:, :, 0].view(-1)
    py = floored_points[:, :, 1].view(-1)
    px = torch.clip(px, 0, height_samples.shape[0] - 2)
    py = torch.clip(py, 0, height_samples.shape[1] - 2)

    # Calculate the fractional part of the points' positions
    fx = points[:, :, 0].view(-1) - px.float()
    fy = points[:, :, 1].view(-1) - py.float()

    # Get the heights of the four surrounding points
    h_tl = height_samples[px, py]  # Top-left
    h_tr = height_samples[px + 1, py]  # Top-right
    h_bl = height_samples[px, py + 1]  # Bottom-left
    h_br = height_samples[px + 1, py + 1]  # Bottom-right

    # Perform bilinear interpolation
    h_t = h_tl + (h_tr - h_tl) * fx  # Top interpolation
    h_b = h_bl + (h_br - h_bl) * fx  # Bottom interpolation
    interpolated_heights = h_t + (h_b - h_t) * fy  # Final interpolation

    # heights = torch.min(heights1, heights2).view(num_envs, -1)
    heights = base_pos[:, 2:3] - interpolated_heights.view(num_envs, -1)

    if return_all_dims:
        # This is only for visualization purposes, plotting the height map the humanoid sees
        points = rotations.quat_apply_yaw(
            base_rot.repeat(1, num_height_points), height_points, w_last
        ) + (base_pos[:, :3]).unsqueeze(1)
        heights = interpolated_heights.view(num_envs, -1, 1)
        return torch.cat(
            [points[..., :2].view(num_envs, -1, 2), heights], dim=-1
        ).clone()

    return heights.view(num_envs, -1).clone()


def convert_heightfield_to_trimesh(
    height_field_raw,
    horizontal_scale,
    vertical_scale,
    slope_threshold=None,
    flat_tolerance=None,
    max_triangle_size=None,
):
    """
    Convert a heightfield array to a triangle mesh with optional slope correction and flat region decimation.

    Args:
        height_field_raw (np.array): Input heightfield
        horizontal_scale (float): Horizontal scale in meters
        vertical_scale (float): Vertical scale in meters
        slope_threshold (float, optional): Slope threshold for surface correction
        flat_tolerance (float, optional): Height tolerance for merging flat regions
        max_triangle_size (float, optional): Maximum triangle size for merged regions

    Returns:
        tuple: (vertices, triangles) arrays defining the mesh
    """
    hf = height_field_raw
    num_rows, num_cols = hf.shape

    print(
        f"Initial heightfield size: {num_rows}x{num_cols} = {num_rows * num_cols} vertices"
    )

    # Create coordinate grids
    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    # Apply slope correction if threshold is provided
    if slope_threshold is not None:
        xx, yy = _correct_slopes(
            hf, xx, yy, horizontal_scale, vertical_scale, slope_threshold
        )

    # Generate basic mesh if no flat tolerance specified
    if flat_tolerance is None:
        return _generate_basic_mesh(xx, yy, hf, vertical_scale, num_rows, num_cols)

    print("Optimizing mesh by merging flat regions...")
    vertices, triangles = _generate_optimized_mesh(
        xx, yy, hf, vertical_scale, horizontal_scale, flat_tolerance, max_triangle_size
    )
    print(f"Optimized mesh vertex count: {len(vertices)}")
    return vertices, triangles


def _correct_slopes(hf, xx, yy, horizontal_scale, vertical_scale, slope_threshold):
    """Apply slope correction to the heightfield coordinates."""
    num_rows, num_cols = hf.shape
    slope_threshold *= horizontal_scale / vertical_scale

    # Calculate movement masks
    move_x = np.zeros((num_rows, num_cols))
    move_y = np.zeros((num_rows, num_cols))
    move_corners = np.zeros((num_rows, num_cols))

    # X direction slopes
    move_x[:-1, :] += hf[1:, :] - hf[:-1, :] > slope_threshold
    move_x[1:, :] -= hf[:-1, :] - hf[1:, :] > slope_threshold

    # Y direction slopes
    move_y[:, :-1] += hf[:, 1:] - hf[:, :-1] > slope_threshold
    move_y[:, 1:] -= hf[:, :-1] - hf[:, 1:] > slope_threshold

    # Corner slopes
    move_corners[:-1, :-1] += hf[1:, 1:] - hf[:-1, :-1] > slope_threshold
    move_corners[1:, 1:] -= hf[:-1, :-1] - hf[1:, 1:] > slope_threshold

    # Apply corrections
    xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
    yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale

    return xx, yy


def _generate_basic_mesh(xx, yy, hf, vertical_scale, num_rows, num_cols):
    """Generate a basic mesh without optimization."""
    vertices = np.column_stack(
        (xx.flatten(), yy.flatten(), hf.flatten() * vertical_scale)
    ).astype(np.float32)

    triangles = []
    for i in range(num_rows - 1):
        for j in range(num_cols - 1):
            base_idx = i * num_cols + j
            triangles.extend(
                [
                    [base_idx, base_idx + num_cols + 1, base_idx + 1],
                    [base_idx, base_idx + num_cols, base_idx + num_cols + 1],
                ]
            )

    return vertices, np.array(triangles, dtype=np.uint32)


def _generate_optimized_mesh(
    xx, yy, hf, vertical_scale, horizontal_scale, flat_tolerance, max_triangle_size
):
    """Generate an optimized mesh with merged flat regions."""
    H = hf * vertical_scale
    rows, cols = H.shape

    # Calculate cell properties
    cell_heights = np.stack([H[:-1, :-1], H[1:, :-1], H[:-1, 1:], H[1:, 1:]])
    cell_avg = np.mean(cell_heights, axis=0)
    cell_diff = np.max(cell_heights, axis=0) - np.min(cell_heights, axis=0)

    # Determine mergeable regions
    merge_margin = int(np.ceil(0.1 / horizontal_scale))
    mergeable = cell_diff <= flat_tolerance
    non_mergeable = ndimage.binary_dilation(~mergeable, iterations=merge_margin)
    allowed_merge = ~non_mergeable

    # Initialize mesh generation
    visited = np.zeros(cell_avg.shape, dtype=bool)
    vertex_map = {}
    vertices_list = []
    faces_list = []

    def add_vertex(i, j):
        key = (i, j)
        if key not in vertex_map:
            vertex_map[key] = len(vertices_list)
            vertices_list.append([xx[i, j], yy[i, j], H[i, j]])
        return vertex_map[key]

    def add_quad(i, j, height, width):
        tl, tr = add_vertex(i, j), add_vertex(i, j + width)
        bl, br = add_vertex(i + height, j), add_vertex(i + height, j + width)
        faces_list.extend([[tl, br, tr], [tl, bl, br]])

    # Process each cell
    for i in range(rows - 1):
        for j in range(cols - 1):
            if visited[i, j]:
                continue

            if not allowed_merge[i, j] or cell_diff[i, j] > flat_tolerance:
                add_quad(i, j, 1, 1)
                visited[i, j] = True
                continue

            # Find mergeable region
            width = height = 1
            base_height = cell_avg[i, j]

            while (
                j + width < cols - 1
                and not visited[i, j + width]
                and allowed_merge[i, j + width]
                and abs(cell_avg[i, j + width] - base_height) <= flat_tolerance
            ):
                width += 1

            while i + height < rows - 1:
                if (
                    not all(~visited[i + height, j : j + width])
                    or not all(allowed_merge[i + height, j : j + width])
                    or not all(
                        abs(cell_avg[i + height, j : j + width] - base_height)
                        <= flat_tolerance
                    )
                ):
                    break
                height += 1

            # Mark region as visited
            visited[i : i + height, j : j + width] = True

            # Add subdivided or single quad
            if max_triangle_size is not None:
                physical_size = np.array([width, height]) * horizontal_scale
                if np.linalg.norm(physical_size) > max_triangle_size:
                    subdivs = np.ceil(physical_size / max_triangle_size).astype(int)
                    for si in range(subdivs[1]):
                        for sj in range(subdivs[0]):
                            sub_height = height // subdivs[1]
                            sub_width = width // subdivs[0]
                            add_quad(
                                i + si * sub_height,
                                j + sj * sub_width,
                                sub_height,
                                sub_width,
                            )
                    continue

            add_quad(i, j, height, width)

    return np.array(vertices_list, dtype=np.float32), np.array(
        faces_list, dtype=np.uint32
    )


def perlin(x, y, seed=0):
    # permutation table
    # np.random.seed(seed)
    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    # coordinates of the top-left
    xi, yi = x.astype(int), y.astype(int)
    # internal coordinates
    xf, yf = x - xi, y - yi
    # fade factors
    u, v = fade(xf), fade(yf)
    # noise components
    n00 = gradient(p[p[xi] + yi], xf, yf)
    n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
    n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)
    # combine noises
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)  # FIX1: I was using n10 instead of n01
    return lerp(x1, x2, v)  # FIX2: I also had to reverse x1 and x2 here


def lerp(a, b, x):
    "linear interpolation"
    return a + x * (b - a)


def fade(t):
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def gradient(h, x, y):
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    g = vectors[h % 4]
    return g[:, :, 0] * x + g[:, :, 1] * y
