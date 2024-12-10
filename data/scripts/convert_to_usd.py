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

import asyncio
from isaacsim import SimulationApp

import os
from pathlib import Path

import numpy as np
import torch
import typer
from tqdm import tqdm


async def convert(in_file, out_file, load_materials=False):
    # This import causes conflicts when global
    import omni.kit.asset_converter

    def progress_callback(progress, total_steps):
        pass

    converter_context = omni.kit.asset_converter.AssetConverterContext()
    # setup converter and flags
    converter_context.ignore_materials = not load_materials
    # converter_context.ignore_animation = False
    # converter_context.ignore_cameras = True
    # converter_context.single_mesh = True
    # converter_context.smooth_normals = True
    # converter_context.preview_surface = False
    # converter_context.support_point_instancer = False
    # converter_context.embed_mdl_in_usd = False
    # converter_context.use_meter_as_world_unit = True
    # converter_context.create_world_as_default_root_prim = False
    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(in_file, out_file, progress_callback, converter_context)
    success = True
    while True:
        success = await task.wait_until_finished()
        if not success:
            await asyncio.sleep(0.1)
        else:
            break
    return success


def main(
    assets_root_dir: Path,
    force_remake: bool = False,
):
    kit = SimulationApp()

    import omni
    from omni.isaac.core.utils.extensions import enable_extension

    enable_extension("omni.kit.asset_converter")
    
    folder_names = [
        f.path.split("/")[-1] for f in os.scandir(assets_root_dir) if f.is_dir()
    ]

    for folder_name in folder_names:
        data_dir = assets_root_dir / folder_name

        print(f"Processing subset {folder_name}")

        files = [
            f
            for f in Path(data_dir).glob("**/*")
        ]
        print(f"Processing {len(files)} files")

        files.sort()

        for filename in tqdm(files):
            if str(filename).endswith(".obj") or str(filename).endswith(".ply"):
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
                status = asyncio.get_event_loop().run_until_complete(
                    convert(str(filename), str(outpath), False)
                )
                if not status:
                    print(f"Failed to convert {filename}")
                    exit(0)
            
    kit.close()


if __name__ == "__main__":
    with torch.no_grad():
        typer.run(main)
