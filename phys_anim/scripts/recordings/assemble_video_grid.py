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
Given pairs of recordings recorded from motion capture
and downstream agents, assembles all of them into a single
video displaying all recordings in a grid.
"""

from multiprocessing import Pool, cpu_count
from functools import partial
from pathlib import Path
import typer
from typing import List, Optional
import cv2
import numpy as np

from tqdm import tqdm
import math

EXT = "webm"
SPACER_PX = 10


def read_into_frames(path: Path, label: str = None):
    """
    Returns stack of numpy frames from video at path.
    """
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (100, 80)
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 3

    vid = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, frame = vid.read()
        if not ret:
            break

        if label is None:
            text = str(path).split("/")[-1].split("_")[0]
        else:
            text = label

        frame = cv2.putText(frame, text, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)

        frames.append(frame)

    return np.stack(frames, axis=0)


def get_vid_fps(path: Path):
    """
    Returns fps of video at path.
    """
    vid = cv2.VideoCapture(str(path))
    return vid.get(cv2.CAP_PROP_FPS)


def write_frames(frames: List[np.ndarray], path: Path, fps: int):
    """
    Writes frames to video at path.
    """
    height = frames[0].shape[0]
    width = frames[0].shape[1]

    fourcc = cv2.VideoWriter_fourcc(*"vp80")

    video = cv2.VideoWriter(
        str(path),
        fourcc,
        fps,
        (width, height),
    )
    for img in frames:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(img)

    video.release()


def cat_frames(frame_a: np.ndarray, frame_b: np.ndarray):
    """
    Concatenates two frames side by side.
    """
    height, width = frame_a.shape[:2]
    output = np.zeros((height, 2 * width, 3), dtype=np.uint8)
    output[:, :width, :] = frame_a
    output[:, width:, :] = frame_b
    return output


def resize_frame(frame: np.ndarray, height: int, width: int):
    """
    Resizes frame to given height and width.
    """
    return cv2.resize(frame, (width, height))


def squarify_frames(frames: np.ndarray):
    """
    Does a centre crop on a batch of frames,
    so that height=width
    """
    height, width = frames.shape[1:3]
    if height == width:
        return frames

    if height > width:
        diff = height - width
        start = diff // 2
        end = start + width
        return frames[:, start:end, :, :]
    else:
        diff = width - height
        start = diff // 2
        end = start + height
        return frames[:, :, start:end, :]


def create_single_grid(
        tupl_input,
        pairs_per_row,
        height,
        width,
        vids_a,
        label_a,
        vids_b,
        label_b,
        max_videos_per_grid,
        output_num_frames,
        output_dir,
        output_ext,
        fps
):
    (
        output_file_idx,
        num_rows,
        generated_videos,
        remaining_videos
    ) = tupl_input

    output_height = num_rows * height + SPACER_PX * (num_rows - 1)
    output_width = pairs_per_row * width * 2 + SPACER_PX * (pairs_per_row - 1)

    raw_frames_a = [read_into_frames(v, label_a) for idx, v in
                    enumerate(vids_a[generated_videos:generated_videos + min(remaining_videos, max_videos_per_grid)])]
    raw_frames_b = [read_into_frames(v, label_b) for idx, v in
                    enumerate(vids_b[generated_videos:generated_videos + min(remaining_videos, max_videos_per_grid)])]

    frames_a = [squarify_frames(f) for f in raw_frames_a]
    frames_b = [squarify_frames(f) for f in raw_frames_b]

    output_frames = []

    for i in range(output_num_frames):
        output_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)

        for j in range(len(frames_a)):
            raw_frame_a = frames_a[j][i % len(frames_a[j])]
            raw_frame_b = frames_b[j][i % len(frames_b[j])]

            frame_a = resize_frame(
                raw_frame_a,
                height,
                width,
            )
            frame_b = resize_frame(
                raw_frame_b,
                height,
                width,
            )

            row_position = j // pairs_per_row
            col_position = j % pairs_per_row

            cat_frame = cat_frames(frame_a, frame_b)

            start_height = row_position * (height + SPACER_PX)
            end_height = start_height + height
            start_width = col_position * (width * 2 + SPACER_PX)
            end_width = start_width + width * 2

            output_frame[start_height:end_height, start_width:end_width, :] = cat_frame

        output_frames.append(output_frame)

    write_frames(output_frames, output_dir / f"output_{output_file_idx}.{output_ext}", fps)


def make_merged_video(
    vids_a: List[Path],
    label_a: Optional[str],
    vids_b: List[Path],
    label_b: Optional[str],
    output_dir: Path,
    output_ext: str,
    output_length_seconds: int,
    pairs_per_row: int,
    rows_per_video: int,
    downscale_output_factor: int,
    num_workers: int
):
    """
    Reads all the input recordings and makes a single combined video where
    corresponding vids are side by side in a grid. Each sub-video in the
    grid loops for the duration of the output video.

    This is done by reading all the input vids and assembling the grid
    video frame by frame.
    """

    dummy_vid = squarify_frames(read_into_frames(vids_a[0]))

    raw_height, raw_width = dummy_vid[0].shape[:2]
    fps = get_vid_fps(vids_a[0])

    height = raw_height // downscale_output_factor
    width = raw_width // downscale_output_factor

    output_num_frames = int(output_length_seconds * fps)

    total_num_rows = math.ceil(len(vids_a) / pairs_per_row)
    remaining_rows = total_num_rows
    num_files = math.ceil(total_num_rows / rows_per_video)

    max_videos_per_grid = rows_per_video * pairs_per_row
    remaining_videos = len(vids_a)
    generated_videos = 0

    list_of_calls = []

    print(f"Generating {num_files} grid files.")
    for output_file_idx in range(num_files):

        num_rows = min(remaining_rows, rows_per_video)

        list_of_calls.append(
            (
                output_file_idx,
                num_rows,
                generated_videos,
                remaining_videos
            )
        )

        remaining_rows -= rows_per_video
        generated_videos += max_videos_per_grid
        remaining_videos -= max_videos_per_grid

    pool = Pool(num_workers)
    func = partial(
        create_single_grid,
        pairs_per_row=pairs_per_row, height=height, width=width, vids_a=vids_a, label_a=label_a, vids_b=vids_b,
        label_b=label_b, max_videos_per_grid=max_videos_per_grid, output_num_frames=output_num_frames,
        output_dir=output_dir, output_ext=output_ext, fps=fps
    )

    with tqdm(total=len(list_of_calls)) as pbar:
        for _ in pool.imap_unordered(func, list_of_calls):
            pbar.update()


DEFAULT_OUTPUT_LENGTH_SECONDS = 10
DEFAULT_PAIRS_PER_ROW = 4
DEFAULT_ROWS_PER_VIDEO = 4
DEFAULT_DOWNSCALE_OUTPUT_FACTOR = 1


def main(
        expert_dir: Path,
        downstream_dir: Path,
        output_dir: Path,
        output_ext: str,
        output_length_seconds: int = DEFAULT_OUTPUT_LENGTH_SECONDS,
        pairs_per_row: int = DEFAULT_PAIRS_PER_ROW,
        rows_per_video: int = DEFAULT_ROWS_PER_VIDEO,
        downscale_output_factor: int = DEFAULT_DOWNSCALE_OUTPUT_FACTOR,
        num_workers: Optional[int] = None,
):
    expert_videos = list(sorted(expert_dir.rglob(f"*{EXT}"), key=lambda x: int(str(x).split("/")[-1].split("_")[0])))
    downstream_videos = list(sorted(downstream_dir.rglob(f"*{EXT}"), key=lambda x: int(str(x).split("/")[-1].split("_")[0])))

    expert_names = [v.name for v in expert_videos]
    downstream_names = [v.name for v in downstream_videos if v.name in expert_names]

    assert expert_names == downstream_names, "Expert recordings must be contained within the downstream recordings"

    expert_videos = [expert_vid for expert_vid in expert_videos if expert_vid.name in expert_names]

    if num_workers is None:
        num_workers = cpu_count()

    make_merged_video(
        vids_a=expert_videos,
        label_a=None,
        vids_b=downstream_videos,
        label_b="policy",
        output_dir=output_dir,
        output_ext=output_ext,
        output_length_seconds=output_length_seconds,
        pairs_per_row=pairs_per_row,
        rows_per_video=rows_per_video,
        downscale_output_factor=downscale_output_factor,
        num_workers=num_workers
    )


if __name__ == "__main__":
    typer.run(main)
