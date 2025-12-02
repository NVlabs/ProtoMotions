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
import math

import numpy as np
from scipy import interpolate

from protomotions.components.terrains.subterrain import SubTerrain
from protomotions.components.terrains.shape_utils import (
    draw_circle,
    draw_curve,
    draw_disk,
    draw_ellipse,
    draw_polygon,
)


def random_uniform_subterrain(
    subterrain: SubTerrain,
    min_height,
    max_height,
    step=1,
    downsampled_scale=None,
):
    """
    Generate a uniform noise terrain

    Parameters
        terrain (SubTerrain): the terrain
        min_height (float): the minimum height of the terrain [meters]
        max_height (float): the maximum height of the terrain [meters]
        step (float): minimum height change between two points [meters]
        downsampled_scale (float): distance between two randomly sampled points ( musty be larger or equal to terrain.horizontal_scale)

    """
    if downsampled_scale is None:
        downsampled_scale = subterrain.horizontal_scale

    # switch parameters to discrete units
    min_height = int(min_height / subterrain.vertical_scale)
    max_height = int(max_height / subterrain.vertical_scale)
    step = int(step / subterrain.vertical_scale)

    heights_range = np.arange(min_height, max_height + step, step)
    height_field_downsampled = np.random.choice(
        heights_range,
        (
            int(subterrain.width * subterrain.horizontal_scale / downsampled_scale),
            int(subterrain.length * subterrain.horizontal_scale / downsampled_scale),
        ),
    )

    x = np.linspace(
        0,
        subterrain.width * subterrain.horizontal_scale,
        height_field_downsampled.shape[0],
    )
    y = np.linspace(
        0,
        subterrain.length * subterrain.horizontal_scale,
        height_field_downsampled.shape[1],
    )

    # Use RegularGridInterpolator instead of interp2d
    f = interpolate.RegularGridInterpolator(
        (x, y), height_field_downsampled, method="linear"
    )

    x_upsampled = np.linspace(
        0, subterrain.width * subterrain.horizontal_scale, subterrain.width
    )
    y_upsampled = np.linspace(
        0, subterrain.length * subterrain.horizontal_scale, subterrain.length
    )

    # Create a grid of points for interpolation
    points = np.meshgrid(x_upsampled, y_upsampled, indexing="ij")
    z_upsampled = np.rint(
        f(np.stack([points[0].flatten(), points[1].flatten()], axis=-1)).reshape(
            subterrain.width, subterrain.length
        )
    )

    subterrain.height_field_raw += z_upsampled.astype(np.int16)

    subterrain.terrain_name = "random_uniform"

    return subterrain


def sloped_subterrain(subterrain: SubTerrain, slope=1):
    """
    Generate a sloped terrain

    Parameters:
        subterrain (protomotions.components.terrains.route_subterrain.RouteSubTerrain): the terrain
        slope (float): positive or negative slope
    Returns:
        terrain (SubTerrain): update terrain
    """

    x = np.arange(0, subterrain.width)
    y = np.arange(0, subterrain.length)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = xx.reshape(subterrain.width, 1)
    max_height = int(
        slope
        * (subterrain.horizontal_scale / subterrain.vertical_scale)
        * subterrain.width
    )
    subterrain.height_field_raw[:, np.arange(subterrain.length)] += (
        max_height * xx / subterrain.width
    ).astype(subterrain.height_field_raw.dtype)

    subterrain.terrain_name = "sloped"

    return subterrain


def pyramid_sloped_subterrain(subterrain: SubTerrain, slope=1, platform_size=1.0):
    """
    Generate a sloped terrain

    Parameters:
        subterrain (terrain): the terrain
        slope (int): positive or negative slope
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    x = np.arange(0, subterrain.width)
    y = np.arange(0, subterrain.length)
    center_x = int(subterrain.width / 2)
    center_y = int(subterrain.length / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = (center_x - np.abs(center_x - xx)) / center_x
    yy = (center_y - np.abs(center_y - yy)) / center_y
    xx = xx.reshape(subterrain.width, 1)
    yy = yy.reshape(1, subterrain.length)
    max_height = int(
        slope
        * (subterrain.horizontal_scale / subterrain.vertical_scale)
        * (subterrain.width / 2)
    )
    subterrain.height_field_raw += (max_height * xx * yy).astype(
        subterrain.height_field_raw.dtype
    )

    platform_size = int(platform_size / subterrain.horizontal_scale / 2)
    x1 = subterrain.width // 2 - platform_size
    y1 = subterrain.length // 2 - platform_size

    min_h = min(subterrain.height_field_raw[x1, y1], 0)
    max_h = max(subterrain.height_field_raw[x1, y1], 0)
    subterrain.height_field_raw = np.clip(subterrain.height_field_raw, min_h, max_h)

    subterrain.terrain_name = "pyramid_sloped"

    return subterrain


def discrete_obstacles_subterrain(
    subterrain: SubTerrain, max_height, min_size, max_size, num_rects, platform_size=1.0
):
    """
    Generate a terrain with gaps

    Parameters:
        subterrain (terrain): the terrain
        max_height (float): maximum height of the obstacles (range=[-max, -max/2, max/2, max]) [meters]
        min_size (float): minimum size of a rectangle obstacle [meters]
        max_size (float): maximum size of a rectangle obstacle [meters]
        num_rects (int): number of randomly generated obstacles
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    subterrain.height_field_raw[:] = 0
    max_height = int(max_height / subterrain.vertical_scale)
    min_size = int(min_size / subterrain.horizontal_scale)
    max_size = int(max_size / subterrain.horizontal_scale)
    platform_size = int(platform_size / subterrain.horizontal_scale)

    (i, j) = subterrain.height_field_raw.shape
    width_range = range(min_size, max_size, 4)
    length_range = range(min_size, max_size, 4)

    for _ in range(num_rects):
        width = np.random.choice(width_range)
        length = np.random.choice(length_range)
        start_i = np.random.choice(range(0, i - width, 4))
        start_j = np.random.choice(range(0, j - length, 4))
        subterrain.height_field_raw[
            start_i : start_i + width, start_j : start_j + length
        ] = max_height  # np.random.choice(height_range)

    x1 = (subterrain.width - platform_size) // 2
    x2 = (subterrain.width + platform_size) // 2
    y1 = (subterrain.length - platform_size) // 2
    y2 = (subterrain.length + platform_size) // 2
    subterrain.height_field_raw[x1:x2, y1:y2] = 0

    subterrain.terrain_name = "discrete_obstacles"

    return subterrain


def obstacles_from_json(subterrain, json_file):
    import json

    subterrain.height_field_raw[:] = 0
    map_description = json.load(open(json_file, "r"))
    update_segmentation(map_description, subterrain)
    update_terrain(map_description, subterrain)
    update_static_obstacles(map_description, subterrain)
    update_top_obstacles(map_description, subterrain)
    update_dynamic_obstacles(subterrain)
    return subterrain


def update_dynamic_obstacles(subterrain):
    for obst in subterrain.dynamic_obstacles:
        obst["end_x"] = obst["start_x"] + obst["cycle"] * obst["velocity_x"]
        obst["end_y"] = obst["start_y"] + obst["cycle"] * obst["velocity_y"]
        obst["cur_pos"] = [obst["start_x"], obst["start_y"], 10]
        obst["cur_vx"] = obst["velocity_x"]
        obst["cur_vy"] = obst["velocity_y"]


def update_top_obstacles(map_description, subterrain):
    slack = 15
    for obst in subterrain.top_obstacles:
        start_x = (
            int(
                (obst.get("cx") - (obst.get("length", 1.0) / 2))
                / subterrain.horizontal_scale
            )
            - slack
        )
        end_x = (
            int(
                (obst.get("cx") + (obst.get("length", 1.0) / 2))
                / subterrain.horizontal_scale
            )
            + slack
        )
        start_y = (
            int(
                (obst.get("cy") - (obst.get("width", 1.0) / 2))
                / subterrain.horizontal_scale
            )
            - slack
        )
        end_y = (
            int(
                (obst.get("cy") + (obst.get("width", 1.0) / 2))
                / subterrain.horizontal_scale
            )
            + slack
        )
        ceiling = int(obst.get("z_bottom") / subterrain.vertical_scale)
        start_x = np.clip(start_x, 0, subterrain.ceiling_field_raw.shape[0])
        end_x = np.clip(end_x, 0, subterrain.ceiling_field_raw.shape[0])
        start_y = np.clip(start_y, 0, subterrain.ceiling_field_raw.shape[1])
        end_y = np.clip(end_y, 0, subterrain.ceiling_field_raw.shape[1])
        subterrain.ceiling_field_raw[start_x:end_x, start_y:end_y] = np.minimum(
            subterrain.ceiling_field_raw[start_x:end_x, start_y:end_y], ceiling
        )
    subterrain.dynamic_obstacles = map_description["dynamic_obstacles"]


def update_static_obstacles(map_description, subterrain):
    subterrain.static_obstacles = map_description["static_obstacles"]
    for obst in subterrain.static_obstacles:
        if obst["type"] == "box":
            start_i = int(obst["x"] / subterrain.horizontal_scale)
            start_j = int(obst["y"] / subterrain.horizontal_scale)
            obs_size = int(obst.get("obs_size", 1) / subterrain.horizontal_scale)
            obs_height = int(obst.get("obs_height", 2) / subterrain.vertical_scale)
            subterrain.height_field_raw[
                start_i : start_i + obs_size, start_j : start_j + obs_size
            ] = obs_height
            subterrain.walkable_field_raw[
                start_i : start_i + obs_size, start_j : start_j + obs_size
            ] = 1
    subterrain.top_obstacles = map_description["top_obstacles"]


def update_terrain(map_description, subterrain):
    if "terrain" in map_description:
        for terrain_desc in map_description["terrain"]:
            start_i = int(terrain_desc.get("start_x", 0) / subterrain.horizontal_scale)
            start_j = int(terrain_desc.get("start_y", 0) / subterrain.horizontal_scale)
            end_i = int(
                terrain_desc.get("end_x", subterrain.width)
                / subterrain.horizontal_scale
            )
            end_j = int(
                terrain_desc.get("end_y", subterrain.length)
                / subterrain.horizontal_scale
            )
            if terrain_desc["type"] == "gravel":
                amplitude = terrain_desc.get("amplitude", 0.05)
                width = end_i - start_i
                length = end_j - start_j
                gravel = np.int16(
                    (2 * amplitude * (np.random.random((width, length)) - 0.5))
                    / subterrain.vertical_scale
                )
                subterrain.height_field_raw[start_i:end_i, start_j:end_j] += gravel
            if terrain_desc["type"] == "sloped":
                sloped_subterrain(subterrain, slope=0.15)
                tmp_hf = subterrain.height_field_raw.copy()
                tmp_hf[(tmp_hf.shape[0] // 2) :, :] = tmp_hf[
                    ((tmp_hf.shape[0] // 2) - 1) :: -1, :
                ]
                subterrain.height_field_raw[:] = tmp_hf
            if terrain_desc["type"] == "stairs":
                stairs_subterrain(subterrain, step_width=0.5, step_height=0.075)
                tmp_hf = subterrain.height_field_raw.copy()
                tmp_hf[(tmp_hf.shape[0] // 2) :, :] = tmp_hf[
                    ((tmp_hf.shape[0] // 2) - 1) :: -1, :
                ]
                subterrain.height_field_raw[:] = tmp_hf
            if terrain_desc["type"] == "mixed":
                vanilla_hf = subterrain.height_field_raw.copy()
                subterrain_stairs = stairs_subterrain(
                    subterrain, step_width=0.5, step_height=0.075
                )
                stairs_hf = subterrain_stairs.height_field_raw
                subterrain.height_field_raw = vanilla_hf
                subterrain_slope = sloped_subterrain(subterrain, slope=0.15)
                slope_hf = subterrain_slope.height_field_raw
                subterrain.height_field_raw[(slope_hf.shape[0] // 2) :, :] = stairs_hf[
                    ((slope_hf.shape[0] // 2) - 1) :: -1, :
                ]
                subterrain.height_field_raw[: (slope_hf.shape[0] // 2), :] = slope_hf[
                    : (slope_hf.shape[0] // 2), :
                ]
                # add gravel
                amplitude = 0.04
                start_i = 0
                end_i = slope_hf.shape[0] // 2
                start_j = slope_hf.shape[1] // 2
                end_j = slope_hf.shape[1]
                width = end_i - start_i
                length = end_j - start_j
                gravel = np.int16(
                    (2 * amplitude * (np.random.random((width, length)) - 0.5))
                    / subterrain.vertical_scale
                )
                subterrain.height_field_raw[start_i:end_i, start_j:end_j] += gravel


def update_segmentation(map_description, subterrain):
    if "segmentation" in map_description:
        segmentation = map_description["segmentation"]
        subterrain.seg_color = {
            seg["name"]: seg.get("color", "none") for seg in segmentation
        }
        subterrain.landmarks = list(np.unique([seg["name"] for seg in segmentation]))
        for x in range(subterrain.length):
            for y in range(subterrain.width):
                min_dist = math.inf
                is_goal = True
                seg_name = None
                for seg in segmentation:
                    cx = seg["cx"] / subterrain.horizontal_scale
                    cy = seg["cy"] / subterrain.horizontal_scale
                    radius = seg.get("radius", math.inf) / subterrain.horizontal_scale
                    dist_to_seg = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    if dist_to_seg < min_dist and dist_to_seg < radius:
                        min_dist = dist_to_seg
                        seg_name = seg["name"].lower()
                        is_goal = seg.get("goal_radius") is None or (
                            dist_to_seg
                            < (seg.get("goal_radius") / subterrain.horizontal_scale)
                        )
                if seg_name is None:
                    raise "No default terrain, fix segmentation!"
                subterrain.segmentation_field[(x, y)] = {
                    "name": seg_name,
                    "is_goal": is_goal,
                }


def wave_subterrain(subterrain, num_waves=1, amplitude=1.0):
    """
    Generate a wavy terrain

    Parameters:
        subterrain (terrain): the terrain
        num_waves (int): number of sine waves across the terrain length
    Returns:
        terrain (SubTerrain): update terrain
    """
    amplitude = int(0.5 * amplitude / subterrain.vertical_scale)
    if num_waves > 0:
        div = subterrain.length / (num_waves * np.pi * 2)
        x = np.arange(0, subterrain.width)
        y = np.arange(0, subterrain.length)
        xx, yy = np.meshgrid(x, y, sparse=True)
        xx = xx.reshape(subterrain.width, 1)
        yy = yy.reshape(1, subterrain.length)
        subterrain.height_field_raw += (
            amplitude * np.cos(yy / div) + amplitude * np.sin(xx / div)
        ).astype(subterrain.height_field_raw.dtype)
    return subterrain


def stairs_subterrain(subterrain, step_width, step_height):
    """
    Generate a stairs

    Parameters:
        subterrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float):  the height of the step [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    step_width = int(step_width / subterrain.horizontal_scale)
    step_height = int(step_height / subterrain.vertical_scale)

    num_steps = subterrain.width // step_width
    height = step_height
    for i in range(num_steps):
        subterrain.height_field_raw[i * step_width : (i + 1) * step_width, :] += height
        height += step_height
    return subterrain


def pyramid_stairs_subterrain(subterrain, step_width, step_height, platform_size=1.0):
    """
    Generate stairs

    Parameters:
        subterrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float): the step_height [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    step_width = int(step_width / subterrain.horizontal_scale)
    step_height = int(step_height / subterrain.vertical_scale)
    platform_size = int(platform_size / subterrain.horizontal_scale)

    height = 0
    start_x = 0
    stop_x = subterrain.width
    start_y = 0
    stop_y = subterrain.length
    while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
        start_x += step_width
        stop_x -= step_width
        start_y += step_width
        stop_y -= step_width
        height += step_height
        subterrain.height_field_raw[start_x:stop_x, start_y:stop_y] = height
    return subterrain


def get_walls_status(cell):
    walls = {
        "N": (cell & 0x1) >> 0,
        "E": (cell & 0x2) >> 1,
        "S": (cell & 0x4) >> 2,
        "W": (cell & 0x8) >> 3,
    }
    return walls


def stepping_stones_subterrain(
    subterrain: SubTerrain,
    stone_size,
    stone_distance,
    max_height,
    platform_size=1.0,
    depth=-10,
):
    """
    Generate a stepping stones terrain

    Parameters:
        subterrain (terrain): the terrain
        stone_size (float): horizontal size of the stepping stones [meters]
        stone_distance (float): distance between stones (i.e size of the holes) [meters]
        max_height (float): maximum height of the stones (positive and negative) [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        depth (float): depth of the holes (default=-10.) [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    stone_size = int(stone_size / subterrain.horizontal_scale)
    stone_distance = int(stone_distance / subterrain.horizontal_scale)
    max_height = int(max_height / subterrain.vertical_scale)
    platform_size = int(platform_size / subterrain.horizontal_scale)
    height_range = np.arange(-max_height - 1, max_height, step=1)

    start_x = 0
    start_y = 0
    subterrain.height_field_raw[:, :] = int(depth / subterrain.vertical_scale)
    if subterrain.length >= subterrain.width:
        while start_y < subterrain.length:
            stop_y = min(subterrain.length, start_y + stone_size)
            start_x = np.random.randint(0, stone_size)
            # fill first hole
            stop_x = max(0, start_x - stone_distance)
            subterrain.height_field_raw[0:stop_x, start_y:stop_y] = np.random.choice(
                height_range
            )
            # fill row
            while start_x < subterrain.width:
                stop_x = min(subterrain.width, start_x + stone_size)
                subterrain.height_field_raw[start_x:stop_x, start_y:stop_y] = (
                    np.random.choice(height_range)
                )
                start_x += stone_size + stone_distance
            start_y += stone_size + stone_distance
    elif subterrain.width > subterrain.length:
        while start_x < subterrain.width:
            stop_x = min(subterrain.width, start_x + stone_size)
            start_y = np.random.randint(0, stone_size)
            # fill first hole
            stop_y = max(0, start_y - stone_distance)
            subterrain.height_field_raw[start_x:stop_x, 0:stop_y] = np.random.choice(
                height_range
            )
            # fill column
            while start_y < subterrain.length:
                stop_y = min(subterrain.length, start_y + stone_size)
                subterrain.height_field_raw[start_x:stop_x, start_y:stop_y] = (
                    np.random.choice(height_range)
                )
                start_y += stone_size + stone_distance
            start_x += stone_size + stone_distance

    x1 = (subterrain.width - platform_size) // 2
    x2 = (subterrain.width + platform_size) // 2
    y1 = (subterrain.length - platform_size) // 2
    y2 = (subterrain.length + platform_size) // 2
    subterrain.height_field_raw[x1:x2, y1:y2] = 0

    subterrain.terrain_name = "stepping_stones"

    return subterrain


def poles_subterrain(subterrain: SubTerrain, difficulty=1):
    """
    Generate stairs

    Parameters:
        subterrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float): the step_height [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    start_x = 0
    stop_x = subterrain.width
    start_y = 0
    stop_y = subterrain.length

    img = np.zeros((subterrain.width, subterrain.length), dtype=np.int16)
    base_prob = 1 / 2
    # disk, circle, curve, poly, ellipse
    probs = np.array([0.9, 0, 0.4, 0.5, 0.5]) * (
        (1 - base_prob) * difficulty + base_prob
    )
    low, high = 200, 500
    num_mult = int(stop_x // 80)

    for i in range(len(probs)):
        p = probs[i]
        if i == 0:
            for _ in range(10 * num_mult):
                if np.random.binomial(1, p):
                    img += draw_disk(img_size=subterrain.width, max_r=7) * int(
                        np.random.uniform(low, high)
                    )
        elif i == 1 and np.random.binomial(1, p):
            for _ in range(5 * num_mult):
                if np.random.binomial(1, p):
                    img += draw_circle(img_size=subterrain.width, max_r=5) * int(
                        np.random.uniform(low, high)
                    )
        elif i == 2 and np.random.binomial(1, p):
            for _ in range(3 * num_mult):
                if np.random.binomial(1, p):
                    img += draw_curve(img_size=subterrain.width) * int(
                        np.random.uniform(low, high)
                    )
        elif i == 3 and np.random.binomial(1, p):
            for _ in range(1 * num_mult):
                if np.random.binomial(1, p):
                    img += draw_polygon(img_size=subterrain.width, max_sides=5) * int(
                        np.random.uniform(low, high)
                    )
        elif i == 4 and np.random.binomial(1, p):
            for _ in range(5 * num_mult):
                if np.random.binomial(1, p):
                    img += draw_ellipse(img_size=subterrain.width, max_size=5) * int(
                        np.random.uniform(low, high)
                    )

    subterrain.height_field_raw[start_x:stop_x, start_y:stop_y] = img

    subterrain.terrain_name = "poles"

    return subterrain
