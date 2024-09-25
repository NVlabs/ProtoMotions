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
Evaluates a policy by recording recordings and making a grid
comparing them to the ground truth motion capture.
"""

import typer
import time
from pathlib import Path
import os
import math

from phys_anim.utils.file_utils import load_motions
from phys_anim.scripts.recordings.assemble_video_grid import (
    main as assemble_video_grid,
    DEFAULT_OUTPUT_LENGTH_SECONDS,
    DEFAULT_PAIRS_PER_ROW,
    DEFAULT_DOWNSCALE_OUTPUT_FACTOR,
    DEFAULT_ROWS_PER_VIDEO,
)
from phys_anim.scripts.recordings.make_reference_videos import (
    main as make_reference_videos,
)
from phys_anim.scripts.recordings.convert_frames_to_videos import (
    main as convert_frames_to_videos,
)


def empty_dir(temp_dir: Path):
    assert_and_run(f"rm -rf {temp_dir}")


def assert_and_run(command: str):
    print(f"Running command: '{command}'")
    os.system(command)
    # assert os.system(command) == 0


EXT = "webm"


def main(
    motion_file: Path,
    checkpoint: Path,
    preprocessed_motion_file: Path = None,
    expert_dir: Path = Path("output/reference_videos"),
    output_dir: Path = Path("output/video_grid"),
    num_motions_per_simulation: int = 128,
    output_length_seconds: int = DEFAULT_OUTPUT_LENGTH_SECONDS,
    pairs_per_row: int = DEFAULT_PAIRS_PER_ROW,
    rows_per_video: int = DEFAULT_ROWS_PER_VIDEO,
    downscale_output_factor: int = DEFAULT_DOWNSCALE_OUTPUT_FACTOR,
    eval_opts: str = "",
    extra_args: str = "",
    clean_expert: bool = False,
    clean_policy: bool = False,
    backbone: str = "isaacgym",
    num_motions_to_record: int = None,
    exp: str = "mimic",
    delete_expert_videos_when_done: bool = False,
    delete_policy_videos_when_done: bool = True,
):
    num_workers = 10

    if not expert_dir.exists() or clean_expert:
        print("Expert recordings not found. Generating them now.")
        assert motion_file is not None
        make_reference_videos(
            motion_file=motion_file,
            dummy_checkpoint=checkpoint,
            preprocessed_motion_file=preprocessed_motion_file,
            num_motions_per_simulation=num_motions_per_simulation,
            output_dir=expert_dir,
            extra_args=extra_args,
            eval_opts=eval_opts,
            num_workers=num_workers,
            backbone=backbone,
            num_motions_to_record=num_motions_to_record,
            exp=exp,
        )

    main_video_dir = output_dir / "recordings"

    if not main_video_dir.exists() or clean_policy:
        print("Main recordings not found. Generating them now.")

        main_video_dir.mkdir(parents=True, exist_ok=True)
        temp_video_dir = output_dir / "temp_videos"

        assert motion_file is not None
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
                + f" checkpoint={checkpoint} +max_eval_steps=300"
                + f" num_envs={motions_this_time}"
                + f" record_dir={temp_video_dir} {extra_args}"
                + f" motion_file={preprocessed_motion_file}"
                + f" width=426 height=240 store_raw=True headless=False init_start_prob=1"
                + f" motion_index_offset={recorded_motions_so_far}"
            )

            assert_and_run(run_command)

            recorded_motions_so_far += motions_this_time + unused_motions_per_iteration

            time.sleep(10)

        convert_frames_to_videos(
            paths=list(temp_video_dir.rglob("*.npy")),
            fps=30,
            remove=True,
            num_workers=num_workers,
            resume=True,
        )

        video_files = [
            f.with_suffix(f".{EXT}") for f in list(temp_video_dir.rglob("*.webm"))
        ]

        for video_file in video_files:
            name = "_".join(str(video_file).split(".")[-2].split("/")[-2:])
            video_file.rename(main_video_dir / f"{name}.{EXT}")

        empty_dir(temp_video_dir)

    assemble_video_grid(
        expert_dir=expert_dir,
        downstream_dir=main_video_dir,
        output_dir=output_dir,
        output_ext=EXT,
        output_length_seconds=output_length_seconds,
        pairs_per_row=pairs_per_row,
        rows_per_video=rows_per_video,
        downscale_output_factor=downscale_output_factor,
        num_workers=num_workers,
    )

    if delete_expert_videos_when_done:
        empty_dir(expert_dir)
    if delete_policy_videos_when_done:
        empty_dir(main_video_dir)


if __name__ == "__main__":
    typer.run(main)
