#!/usr/bin/env python3
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
"""Create a checkerboard texture for the ground plane."""

import numpy as np
from PIL import Image
import os


def create_checkerboard_texture(
    output_path,
    size=512,
    checker_size=64,
    light_color=(220, 220, 220),
    dark_color=(80, 80, 80),
):
    """
    Create a checkerboard texture image.

    Args:
        output_path: Path to save the texture
        size: Texture size in pixels (square)
        checker_size: Size of each checker square in pixels
        light_color: RGB color for light squares
        dark_color: RGB color for dark squares
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Create checkerboard pattern
    for i in range(size):
        for j in range(size):
            # Determine which square we're in
            square_i = i // checker_size
            square_j = j // checker_size
            # Alternate between light and dark
            if (square_i + square_j) % 2 == 0:
                img[i, j] = light_color
            else:
                img[i, j] = dark_color

    # Save the texture
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(img).save(output_path)
    print(f"Checkerboard texture created: {output_path}")
    print(f"  Size: {size}x{size} pixels")
    print(f"  Checker size: {checker_size}x{checker_size} pixels")
    print(f"  Light color: {light_color}")
    print(f"  Dark color: {dark_color}")


if __name__ == "__main__":
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_path = os.path.join(
        project_root, "protomotions/data/assets/checkerboard/checkerboard_texture.png"
    )

    create_checkerboard_texture(output_path)
