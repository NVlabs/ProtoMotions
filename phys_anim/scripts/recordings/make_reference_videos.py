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

"""
Given a list of motion ids, records recordings by playing the groundtruth motions.
"""

import math

import typer
import time
from pathlib import Path
import os
from tqdm import tqdm

from phys_anim.utils.file_utils import load_motions
from phys_anim.scripts.recordings.convert_frames_to_videos import (
    main as convert_frames_to_videos,
)

from typing import List


def assert_and_run(command: str):
    print(f"Running command: '{command}'")
    os.system(command)
    # assert os.system(command) == 0


def empty_dir(temp_dir: Path):
    assert_and_run(f"rm -rf {temp_dir}")


EXT = "webm"


def main(
    motion_file: Path,
    dummy_checkpoint: Path,
    preprocessed_motion_file: Path = None,
    num_motions_per_simulation: int = 128,
    output_dir: Path = Path("output/reference_videos"),
    extra_args: str = "",
    eval_opts: str = "",
    num_workers: int = None,
    backbone: str = "isaacgym",
    num_motions_to_record: int = None,
    exp: str = "mimic",
):
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = output_dir / "temp_videos"
    motions = load_motions(motion_file)

    if preprocessed_motion_file is None:
        preprocessed_motion_file = motion_file

    total_num_motions = len(motions)
    if num_motions_to_record is None:
        num_motions_to_record = total_num_motions

    unused_motions = total_num_motions - num_motions_to_record
    unused_motions_per_iteration = math.ceil(
        unused_motions
        * 1.0
        / (num_motions_to_record * 1.0 / num_motions_per_simulation)
    )

    recorded_motions_so_far = 0
    while recorded_motions_so_far < total_num_motions:
        motions_this_time = min(
            total_num_motions - recorded_motions_so_far, num_motions_per_simulation
        )

        run_command = (
            f"python phys_anim/eval_agent.py +exp={exp} +backbone={backbone} +opt=[{eval_opts},record]"
            + f" num_envs={motions_this_time} max_eval_steps=300 sync_motion=True"
            + f" record_dir={temp_dir} {extra_args}"
            + f" motion_file={preprocessed_motion_file}"
            + f" width=426 height=240 store_raw=True"
            + f" checkpoint={dummy_checkpoint} headless=False reset_on_reset_track=False"
            + f" motion_index_offset={recorded_motions_so_far} ref_respawn_offset=0"
        )

        assert_and_run(run_command)

        recorded_motions_so_far += motions_this_time + unused_motions_per_iteration

        time.sleep(10)

    convert_frames_to_videos(
        list(temp_dir.rglob("*.npy")),
        fps=30,
        remove=True,
        num_workers=num_workers,
        resume=True,
    )

    video_files = [f.with_suffix(f".{EXT}") for f in list(temp_dir.rglob("*.webm"))]

    for video_file in video_files:
        name = "_".join(str(video_file).split(".")[-2].split("/")[-2:])
        video_file.rename(output_dir / f"{name}.{EXT}")

    empty_dir(temp_dir)


if __name__ == "__main__":
    typer.run(main)
