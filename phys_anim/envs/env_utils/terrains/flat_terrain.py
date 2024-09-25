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

from phys_anim.envs.env_utils.terrains.terrain import Terrain
from phys_anim.envs.env_utils.terrains.terrain_utils import (
    convert_heightfield_to_trimesh,
)
import numpy as np
import math
from phys_anim.utils.scene_lib import SceneLib


class FlatTerrain(Terrain):
    def __init__(self, config, scene_lib: SceneLib, num_envs: int, device) -> None:
        config.load_terrain = False
        config.save_terrain = False

        env_rows = config.num_levels
        env_cols = config.num_terrains
        num_maps = env_rows * env_cols

        total_size = num_maps * config.map_length * config.map_width * 1.0
        space_between_humanoids = total_size / num_envs
        space_multiplier = max(
            1, config.minimal_humanoid_spacing / space_between_humanoids
        )
        # When creating a simple flat terrain, we need to make sure that there is enough space between the humanoids.
        # For irregular terrains the user will have to ensure that there is enough space between the humanoids.
        config.num_terrains = math.ceil(config.num_terrains * space_multiplier)

        super().__init__(config, scene_lib, num_envs, device)
        # Re-create vertices and triangles for a simple flat mesh
        self.recreate_simple_mesh()

    def generate_subterrains(self):
        self.flat_field_raw[:] = 0
        # Override to do nothing else
        pass

    def recreate_simple_mesh(self):
        # Create a 2x2 height field
        height_field_raw = np.zeros((2, 2), dtype=np.int16)

        # Calculate the new horizontal scale
        resized_horizontal_scale = self.horizontal_scale * (self.tot_cols - 1)

        # Generate new vertices and triangles
        new_vertices, new_triangles = convert_heightfield_to_trimesh(
            height_field_raw,
            resized_horizontal_scale,
            self.vertical_scale,
            None,
        )

        # Assign corner coordinates from original vertices to new vertices
        new_vertices[:, 0] = np.array(
            [
                self.vertices[:, 0].min(),
                self.vertices[:, 0].min(),
                self.vertices[:, 0].max(),
                self.vertices[:, 0].max(),
            ]
        )
        new_vertices[:, 1] = np.array(
            [
                self.vertices[:, 1].min(),
                self.vertices[:, 1].max(),
                self.vertices[:, 1].min(),
                self.vertices[:, 1].max(),
            ]
        )

        # If all assertions pass, overwrite the original vertices and triangles
        self.vertices = new_vertices
        self.triangles = new_triangles
