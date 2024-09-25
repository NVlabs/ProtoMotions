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
from typing import Optional

import numpy as np
import typer
import yaml


def main(
    main_motion_dir: Path,
    humanoid_type: str = "smpl",
    amass_fps_file: Optional[Path] = None,
    output_path: Optional[Path] = None,
):
    if humanoid_type == "smplx":
        assert (
            amass_fps_file is not None
        ), "Please provide the amass_fps_file since amass-x fps is wrong."
        amass_fps = yaml.load(open(amass_fps_file, "r"), Loader=yaml.SafeLoader)

    # iterate over folder and all sub folders recursively.
    # load each file.
    # store the full filename in a dictionary.
    # store the entry "motion_fps" in the dictionary.
    # save the dictionary to a yaml file.
    motion_fps_dict = {}
    for root, dirs, files in os.walk(main_motion_dir):
        # Ignore folders with name "-retarget" or "-smpl" or "-smplx"
        if "-retarget" in root or "-smpl" in root or "-smplx" in root:
            continue
        for file in files:
            if (
                file.endswith(".npz")
                and file != "shape.npz"
                and "stagei.npz" not in file
            ):
                # remove the main_motion_dir from the root
                save_root = root.replace(str(main_motion_dir), "")
                # remove any leading slashes
                save_root = save_root.lstrip("/")

                file_rename = (
                    save_root
                    + "/"
                    + file.replace(".npz", ".npy")
                    .replace("-", "_")
                    .replace(" ", "_")
                    .replace("(", "_")
                    .replace(")", "_")
                )

                if humanoid_type == "smplx":
                    amass_filename = file_rename.replace("_stageii", "_poses")
                    amass_filename = amass_filename.replace("SSM/", "SSM_synced/")
                    amass_filename = amass_filename.replace("HMD05/", "MPI_HDM05/")
                    amass_filename = amass_filename.replace("MoSh/", "MPI_mosh/")
                    amass_filename = amass_filename.replace("PosePrior/", "MPI_Limits/")
                    amass_filename = amass_filename.replace(
                        "TCDHands/", "TCD_handMocap/"
                    )
                    amass_filename = amass_filename.replace(
                        "Transitions/", "Transitions_mocap/"
                    )
                    amass_filename = amass_filename.replace("DFaust/", "DFaust_67/")
                    amass_filename = amass_filename.replace(
                        "BMLrub/", "BioMotionLab_NTroje/"
                    )

                    if amass_filename in amass_fps:
                        framerate = amass_fps[amass_filename]
                    else:
                        motion_data = dict(
                            np.load(open(root + "/" + file, "rb"), allow_pickle=True)
                        )
                        if "TotalCapture" in file_rename or "SSM" in file_rename:
                            framerate = 60
                        elif "KIT" in file_rename:
                            framerate = 100
                        elif "mocap_frame_rate" in motion_data:
                            framerate = motion_data["mocap_frame_rate"]
                        else:
                            raise Exception(f"{file_rename} has no framerate")
                else:
                    motion_data = dict(
                        np.load(open(root + "/" + file, "rb"), allow_pickle=True)
                    )
                    if "mocap_framerate" in motion_data:
                        framerate = motion_data["mocap_framerate"]
                    else:
                        raise Exception(f"{file_rename} has no framerate")

                motion_fps_dict[file_rename] = int(framerate)

    if output_path is None:
        output_path = Path.cwd()
    with open(output_path / f"motion_fps_{humanoid_type}.yaml", "w") as f:
        yaml.dump(motion_fps_dict, f)


if __name__ == "__main__":
    typer.run(main)
