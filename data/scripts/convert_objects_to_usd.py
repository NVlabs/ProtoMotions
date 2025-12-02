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
from isaaclab.app import AppLauncher

import os
from pathlib import Path
import torch
import typer
from tqdm import tqdm


def convert(in_file, out_file, make_instanceable, collision_approximation, mass):
    # Mass properties
    from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
    from isaaclab.sim.schemas import schemas_cfg

    if mass is not None:
        mass_props = schemas_cfg.MassPropertiesCfg(mass=mass)
        rigid_props = schemas_cfg.RigidBodyPropertiesCfg()
    else:
        mass_props = None
        rigid_props = None

    # Collision properties
    collision_props = schemas_cfg.CollisionPropertiesCfg(
        collision_enabled=collision_approximation != "none"
    )

    # Create Mesh converter config
    mesh_converter_cfg = MeshConverterCfg(
        mass_props=mass_props,
        rigid_props=rigid_props,
        collision_props=collision_props,
        asset_path=in_file,
        force_usd_conversion=True,
        usd_dir=os.path.dirname(out_file),
        usd_file_name=os.path.basename(out_file),
        make_instanceable=make_instanceable,
        collision_approximation=collision_approximation,
    )
    MeshConverter(mesh_converter_cfg)


def process_file(
    filename: Path,
    base_dir: Path,
    force_remake: bool,
    make_instanceable: bool,
    collision_approximation: str,
    mass: float | None,
):
    """Processes a single mesh file for conversion to USD."""
    supported_endings = (".obj", ".ply", ".stl")
    if str(filename).endswith(supported_endings):
        # Ensure base_dir is absolute for correct relative_to calculation
        if not base_dir.is_absolute():
            base_dir = base_dir.resolve()
        if not filename.is_absolute():
            filename = filename.resolve()

        try:
            relative_path_dir = filename.relative_to(base_dir).parent
        except ValueError:
            # If filename is not inside base_dir (e.g., single file input case)
            # place the output USD in the same directory as the input file.
            relative_path_dir = filename.parent.relative_to(
                filename.parent
            )  # This results in '.'

        file_ending = filename.name.split(".")[-1]
        # Determine output directory based on whether it's part of a larger scan or single file
        if (
            base_dir == filename.parent
        ):  # Single file case or file directly in assets_root_dir
            output_base = filename.parent
        else:  # File inside a subdirectory scan
            output_base = base_dir

        outpath = (
            output_base / relative_path_dir / filename.name.replace(file_ending, "usd")
        )

        if outpath.exists() and not force_remake:
            # print(f"Skipping existing {outpath}")
            return  # Skip if exists and not forcing remake

        print(f"Processing {filename} -> {outpath}")
        convert(
            str(filename),
            str(outpath),
            make_instanceable,
            collision_approximation,
            mass,
        )


def main(
    assets_root_dir: Path,
    force_remake: bool = False,
    make_instanceable: bool = False,
    collision_approximation: str = "none",
    mass: float = None,
):
    app_launcher = AppLauncher({"headless": True})
    simulation_app = app_launcher.app

    """Rest everything follows."""

    if not assets_root_dir.exists():
        print(f"Error: Provided path {assets_root_dir} does not exist.")
        simulation_app.close()
        return

    if assets_root_dir.is_file():
        print(f"Processing single file: {assets_root_dir}")
        # Use the parent directory as the base for output structure
        process_file(
            assets_root_dir,
            assets_root_dir.parent,  # Base dir is the parent for relative path calculation/output structure
            force_remake,
            make_instanceable,
            collision_approximation,
            mass,
        )
    elif assets_root_dir.is_dir():
        print(f"Processing directory: {assets_root_dir}")
        # Check for subdirectories first
        subdirs = [d for d in assets_root_dir.iterdir() if d.is_dir()]

        if subdirs:
            print(f"Found subdirectories: {[d.name for d in subdirs]}")
            # Process each subdirectory
            for data_dir in subdirs:
                print(f"Processing subset {data_dir.name}")
                files = [f for f in data_dir.glob("**/*") if f.is_file()]
                print(f"Found {len(files)} files in {data_dir.name}")
                files.sort()
                for filename in tqdm(files, desc=f"Converting {data_dir.name}"):
                    process_file(
                        filename,
                        data_dir,  # Base directory for this subset
                        force_remake,
                        make_instanceable,
                        collision_approximation,
                        mass,
                    )
        else:
            # No subdirectories, process files directly in assets_root_dir
            files = [f for f in assets_root_dir.glob("*") if f.is_file()]
            print(
                f"No subdirectories found. Processing {len(files)} files directly in {assets_root_dir}"
            )
            files.sort()
            for filename in tqdm(files, desc=f"Converting {assets_root_dir.name}"):
                process_file(
                    filename,
                    assets_root_dir,  # Base directory for relative path calculation
                    force_remake,
                    make_instanceable,
                    collision_approximation,
                    mass,
                )
    else:
        # This case should theoretically not be reached due to the initial exists() check
        # but kept for robustness.
        print(
            f"Error: Provided path {assets_root_dir} is not a recognized file or directory."
        )
        simulation_app.close()
        return

    simulation_app.close()


if __name__ == "__main__":
    with torch.no_grad():
        typer.run(main)
