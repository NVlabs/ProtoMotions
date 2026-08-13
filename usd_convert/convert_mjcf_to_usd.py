# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utility to convert a MJCF into USD format.

MuJoCo XML Format (MJCF) is an XML file format used in MuJoCo to describe all elements of a robot.
For more information, see: http://www.mujoco.org/book/XMLreference.html

This script uses the MJCF importer extension from Isaac Sim (``isaacsim.asset.importer.mjcf``) to convert
a MJCF asset into USD format. It is designed as a convenience script for command-line use. For more information
on the MJCF importer, see the documentation for the extension:
https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/ext_isaacsim_asset_importer_mjcf.html


positional arguments:
  input               The path to the input MJCF file.
  output_dir          The directory in which to store the converted USD asset.

optional arguments:
  -h, --help                Show this help message and exit
  --fix-base                Fix the base to where it is imported. (default: False)
  --self-collision          Enable self-collisions. (default: False)
  --merge-mesh              Merge meshes where possible. (default: False)
  --collision-from-visuals  Generate collision from visuals. (default: False)

"""

"""Launch Isaac Sim Simulator first."""

import argparse  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Utility to convert a MJCF into USD format."
)
parser.add_argument("input", type=str, help="Path to the input MJCF file.")
parser.add_argument(
    "output_dir",
    type=str,
    help=(
        "Directory for the converted asset. IsaacLab writes "
        "<input-stem>/<input-stem>.usda below this directory."
    ),
)
parser.add_argument(
    "--fix-base",
    action="store_true",
    default=False,
    help="Fix the base to where it is imported.",
)
parser.add_argument(
    "--self-collision",
    action="store_true",
    default=False,
    help="Activate self-collisions between links of the articulation.",
)
parser.add_argument(
    "--merge-mesh",
    action="store_true",
    default=False,
    help="Merge meshes where possible to optimize the model.",
)
parser.add_argument(
    "--collision-from-visuals",
    action="store_true",
    default=False,
    help="Generate collision geometry from visual geometries.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib  # noqa: E402
import os  # noqa: E402

import carb  # noqa: E402
import isaacsim.core.utils.stage as stage_utils  # noqa: E402
import omni.kit.app  # noqa: E402

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg  # noqa: E402
from isaaclab.utils.assets import check_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402


def main():
    # check valid file path
    mjcf_path = args_cli.input
    if not os.path.isabs(mjcf_path):
        mjcf_path = os.path.abspath(mjcf_path)
    if not check_file_path(mjcf_path):
        raise ValueError(f"Invalid file path: {mjcf_path}")
    # create destination directory
    dest_dir = args_cli.output_dir
    if not os.path.isabs(dest_dir):
        dest_dir = os.path.abspath(dest_dir)

    # create the converter configuration (IsaacLab 3 API)
    mjcf_converter_cfg = MjcfConverterCfg(
        asset_path=mjcf_path,
        usd_dir=dest_dir,
        force_usd_conversion=True,
        fix_base=args_cli.fix_base,
        self_collision=args_cli.self_collision,
        merge_mesh=args_cli.merge_mesh,
        collision_from_visuals=args_cli.collision_from_visuals,
    )

    # Print info
    print("-" * 80)
    print("-" * 80)
    print(f"Input MJCF file: {mjcf_path}")
    print("MJCF importer config:")
    print_dict(mjcf_converter_cfg.to_dict(), nesting=0)
    print("-" * 80)
    print("-" * 80)

    # Create mjcf converter and import the file
    mjcf_converter = MjcfConverter(mjcf_converter_cfg)
    # print output
    print("MJCF importer output:")
    print(f"Generated USD file: {mjcf_converter.usd_path}")
    print("-" * 80)
    print("-" * 80)

    # Determine if there is a GUI to update:
    # acquire settings interface
    carb_settings_iface = carb.settings.get_settings()
    # read flag for whether a local GUI is enabled
    local_gui = carb_settings_iface.get("/app/window/enabled")
    # read flag for whether livestreaming GUI is enabled
    livestream_gui = carb_settings_iface.get("/app/livestream/enabled")

    # Simulate scene (if not headless)
    if local_gui or livestream_gui:
        # Open the stage with USD
        stage_utils.open_stage(mjcf_converter.usd_path)
        # Reinitialize the simulation
        app = omni.kit.app.get_app_interface()
        # Run simulation
        with contextlib.suppress(KeyboardInterrupt):
            while app.is_running():
                # perform step
                app.update()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
