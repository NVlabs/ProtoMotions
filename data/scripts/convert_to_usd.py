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

    import contextlib
    import os

    import carb
    import isaacsim.core.utils.stage as stage_utils
    import omni.kit.app

    folder_names = [
        f.path.split("/")[-1] for f in os.scandir(assets_root_dir) if f.is_dir()
    ]

    for folder_name in folder_names:
        data_dir = assets_root_dir / folder_name

        print(f"Processing subset {folder_name}")

        files = [f for f in Path(data_dir).glob("**/*")]
        print(f"Processing {len(files)} files")

        files.sort()

        for filename in tqdm(files):
            if str(filename).endswith(".obj") or str(filename).endswith(".ply") or str(filename).endswith(".stl"):
                relative_path_dir = filename.relative_to(data_dir).parent
                file_ending = filename.name.split(".")[-1]
                outpath = (
                    data_dir
                    / relative_path_dir
                    / filename.name.replace(file_ending, "usda")
                )

                if outpath.exists() and not force_remake:
                    continue

                print(f"Processing {filename}")
                convert(
                    str(filename),
                    str(outpath),
                    make_instanceable,
                    collision_approximation,
                    mass,
                )

    simulation_app.close()


if __name__ == "__main__":
    with torch.no_grad():
        typer.run(main)
