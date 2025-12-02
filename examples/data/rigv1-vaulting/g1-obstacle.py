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
from protomotions.components.scene_lib import (
    Scene,
    BoxSceneObject,
    ObjectOptions,
    SceneLib,
)
import yaml
import re
import argparse
import warnings


class RigV1VaultingAugmented:
    """A class that provides the augmented vaulting scenes for Rig v1."""

    @staticmethod
    def get_scenes(yaml_path):
        # Load configurations from YAML
        with open(yaml_path, "r") as f:
            entry = yaml.safe_load(f)

        # Define the manual box mapping
        vaulting_manual_box_mapping = {
            "generated_vault_over_small_obstacle.motion": (
                1.6 * 0.7,
                0.4 * 0.6,
                0.7 * 0.7,
                (-0.5 * 0.8, 4.8 * 0.9 - 0.25, 0.35 * 0.9),
                (0.0, 0.0, 0.0, 1.0),
            ),
        }

        # Create options (same as in rigv1_vaulting.py)
        object_options = ObjectOptions(
            density=1000,
            fix_base_link=True,
            max_angular_velocity=100.0,
        )

        # Create scenes
        scenes = []

        for motion in entry.get("motions"):
            file_path = motion.get("file")
            motion_id = motion.get("idx")

            # Check if scale_factor exists, use default if not
            if "scale_factor" not in motion:
                warnings.warn(
                    f"Missing scale_factor for motion {file_path}, using default value 1.0"
                )
                scale = 1.0
            else:
                scale = motion.get("scale_factor")

            # Extract the base motion file name to look up in mapping
            # Remove _augN.motion suffix to get the original motion file
            base_motion_file = re.sub(r"_aug\d+\.motion$", ".motion", file_path)

            # Look up the nominal box parameters
            # if base_motion_file in vaulting_manual_box_mapping:
            width, depth, height, translation, rotation = vaulting_manual_box_mapping[
                base_motion_file
            ]

            scaled_height = height * scale

            # Adjust z component of translation to keep the bottom at the same position
            new_translation = list(translation)
            height_diff = scaled_height - height
            new_translation[2] = (
                translation[2] + height_diff / 2
            )  # Adjust z to maintain center position

            objects = []

            o = BoxSceneObject(
                width=width,
                depth=depth,
                height=scaled_height,
                options=object_options,
                translation=tuple(new_translation),
                rotation=rotation,
            )
            objects.append(o)

            scene = Scene(objects=objects, humanoid_motion_id=motion_id)
            scenes.append(scene)

        return scenes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate vaulting scenes from YAML file"
    )
    parser.add_argument(
        "--yaml",
        type=str,
        default="",
        help="Relative path to YAML file from project root",
    )
    parser.add_argument(
        "--output", type=str, default="", help="Path to output scenes file"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    yaml_path = args.yaml
    store_path = args.output

    print(f"Using YAML file: {yaml_path}")
    print(f"Output will be saved to: {store_path}")

    scenes = RigV1VaultingAugmented.get_scenes(yaml_path)

    print(f"Generated {len(scenes)} scenes.")

    # Save scenes using static method (no SceneLib instance needed)
    SceneLib.save_scenes_to_file(scenes, store_path)
    print(f"Saved {len(scenes)} scenes to {store_path}")
    print("To use:")
    print(
        f"  config = SceneLibConfig(scene_file='{store_path}', pointcloud_samples_per_object=8, ...)"
    )
    print(
        "  scene_lib = SceneLib(config, num_envs, scenes=None, device, terrain, scene_weights)"
    )
