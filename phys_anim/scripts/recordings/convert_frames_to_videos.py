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

import typer
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm
from functools import partial
from typing import List, Optional

from phys_anim.agents.callbacks.export_video import write_frames_to_video

SUFFIX = "webm"


def convert_video(source: Path, fps: int, remove: bool, resume: bool):
    dest = source.with_suffix(f".{SUFFIX}")

    try:
        if not dest.exists() or not resume:
            frames = np.load(source)
            write_frames_to_video(frames, dest, fps)

        if remove:
            source.unlink()
    except:
        pass


def main(
        paths: List[Path],
        fps: int = 30,
        remove: bool = True,
        num_workers: Optional[int] = None,
        resume: bool = True,
):
    print(f"Converting {len(paths)} videos to {SUFFIX} format.")

    if num_workers is None:
        num_workers = cpu_count()

    pool = Pool(num_workers)
    func = partial(convert_video, remove=remove, fps=fps, resume=resume)

    with tqdm(total=len(paths)) as pbar:
        for _ in pool.imap_unordered(func, paths):
            pbar.update()

    # results = list(tqdm(pool.imap(func, paths)))


if __name__ == "__main__":
    typer.run(main)
