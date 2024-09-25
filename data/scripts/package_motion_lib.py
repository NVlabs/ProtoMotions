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

from phys_anim.data.assets.skeleton_configs import isaacgym_asset_file_to_stats
from phys_anim.utils.motion_lib import MotionLib


def main(
    motion_file: Path,
    amass_data_path: Path,
    outpath: Path,
    humanoid_type: str = "smpl",
    create_text_embeddings: bool = False,
    num_data_splits: int = None,
):
    if humanoid_type is not None:
        if humanoid_type == "amp_3d":
            # ['right_hand', 'left_hand', 'right_foot', 'left_foot']
            key_body_ids = [5, 8, 11, 14]
            asset_file = "mjcf/amp_humanoid_3d.xml"
        elif humanoid_type == "amp":
            # ['right_hand', 'left_hand', 'right_foot', 'left_foot']
            key_body_ids = [5, 8, 11, 14]
            asset_file = "mjcf/amp_humanoid.xml"
        elif humanoid_type == "amp_sword":
            # ["right_hand", "left_hand", "right_foot", "left_foot", "sword", "shield"]
            key_body_ids = [5, 10, 13, 16, 6, 9]
            asset_file = "mjcf/amp_humanoid_sword_shield.xml"
        elif humanoid_type == "smpl":
            # ["right_hand", "left_hand", "right_foot", "left_foot"]
            key_body_ids = [7, 3, 18, 23]
            asset_file = "mjcf/smpl_humanoid.xml"
        elif humanoid_type == "smplx":
            # ["right_hand", "left_hand", "right_foot", "left_foot"]
            key_body_ids = [7, 3, 17, 36]
            asset_file = "mjcf/smplx_box_humanoid.xml"
        else:
            raise ValueError(f"Unknown humanoid type '{humanoid_type}'")

    (
        _dof_body_ids,
        _dof_offsets,
        _dof_obs_size,
        _num_obs,
        _num_actions,
    ) = isaacgym_asset_file_to_stats(asset_file, len(key_body_ids), True)

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
        with open(motion_file, 'r') as f:
            motion_data = yaml.safe_load(f)
        
        # Edit file paths
        for motion in motion_data['motions']:
            motion['file'] = str(amass_data_path.resolve() / motion['file'])
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
            yaml.dump(motion_data, temp_file)
            temp_file_path = temp_file.name

        # Use the temporary file for MotionLib
        mlib = MotionLib(
            temp_file_path,
            dof_body_ids=_dof_body_ids,
            dof_offsets=_dof_offsets,
            key_body_ids=key_body_ids,
            device="cpu",
            create_text_embeddings=create_text_embeddings,
        )

        print("Saving motion state")

        with open(outpath, "wb") as file:
            torch.save(mlib.state, file)

        # Remove the temporary file
        os.unlink(temp_file_path)


if __name__ == "__main__":
    typer.run(main)
