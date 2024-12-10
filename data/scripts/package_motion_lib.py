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

import os
from pathlib import Path

import torch
import typer
import yaml
import tempfile
from omegaconf import OmegaConf
from hydra.utils import instantiate
from hydra import compose, initialize


def main(
    motion_file: Path,
    amass_data_path: Path,
    outpath: Path,
    humanoid_type: str = "smpl",
    create_text_embeddings: bool = False,
    num_data_splits: int = None,
):
    config_path = "../../phys_anim/config/robot"

    with initialize(version_base=None, config_path=config_path, job_name="test_app"):
        cfg = compose(config_name=humanoid_type)

    key_body_ids = torch.tensor(
        [
            cfg.robot.isaacgym_body_names.index(key_body_name)
            for key_body_name in cfg.robot.key_bodies
        ],
        dtype=torch.long,
    )
    dof_offsets = []
    previous_dof_name = "null"
    for dof_offset, dof_name in enumerate(cfg.robot.isaacgym_dof_names):
        if dof_name[:-2] != previous_dof_name:  # remove the "_x/y/z"
            previous_dof_name = dof_name[:-2]
            dof_offsets.append(dof_offset)
    dof_offsets.append(len(cfg.robot.isaacgym_dof_names))

    print("Creating motion state")
    motion_files = []
    if num_data_splits is not None:
        # Motion file is a yaml file
        # Just load the yaml and break it into num_data_splits
        # Save each split as a separate file
        with open(os.path.join(os.getcwd(), motion_file), "r") as f:
            motions = yaml.load(f, Loader=yaml.SafeLoader)["motions"]
        num_motions = len(motions)
        split_size = num_motions // num_data_splits
        for i in range(num_data_splits):
            if i == num_data_splits - 1:  # make sure we get all the remaining motions
                split_motions = motions[i * split_size :]
            else:
                split_motions = motions[i * split_size : (i + 1) * split_size]

            motion_idx = 0
            for motion in split_motions:
                motion["idx"] = motion_idx
                if "sub_motions" in motion:
                    for sub_motion in motion["sub_motions"]:
                        sub_motion["idx"] = motion_idx
                        motion_idx += 1
                else:
                    motion_idx += 1

            split_name = motion_file.with_name(
                motion_file.stem + f"_{i}" + motion_file.suffix
            )
            with open(split_name, "w") as f:
                yaml.dump({"motions": split_motions}, f)

            motion_files.append(
                (
                    str(split_name),
                    outpath.with_name(outpath.stem + f"_{i}" + outpath.suffix),
                )
            )
    else:
        motion_files.append((motion_file, outpath))

    for motion_file, outpath in motion_files:
        # Open and edit the motion file
        with open(motion_file, "r") as f:
            motion_data = yaml.safe_load(f)

        # Edit file paths
        for motion in motion_data["motions"]:
            motion["file"] = str(amass_data_path.resolve() / motion["file"])

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as temp_file:
            yaml.dump(motion_data, temp_file)
            temp_file_path = temp_file.name

        # Use the temporary file for MotionLib
        cfg.motion_lib.motion_file = temp_file_path
        mlib = instantiate(
            cfg.motion_lib,
            dof_body_ids=cfg.robot.isaacgym_dof_body_ids,
            dof_offsets=dof_offsets,
            key_body_ids=key_body_ids,
            device="cpu",
            create_text_embeddings=create_text_embeddings,
            skeleton_tree=None,
        )

        print("Saving motion state")

        with open(outpath, "wb") as file:
            torch.save(mlib.state, file)

        # Remove the temporary file
        os.unlink(temp_file_path)


if __name__ == "__main__":
    typer.run(main)
