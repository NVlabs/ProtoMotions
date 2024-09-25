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

from phys_anim.agents.callbacks.base_callback import RL_EvalCallback
from phys_anim.agents.ppo import PPO
from phys_anim.envs.humanoid.common import BaseHumanoid

from pathlib import Path

import numpy as np
import cv2

SUFFIX_TO_FOURCC = {
    "webm": "vp80",
    "mp4": "MP4V",
}


def write_frames_to_video(frames: np.ndarray, out: Path, fps: int):
    suffix = out.suffix[1:]
    fourcc = SUFFIX_TO_FOURCC[suffix]
    height, width, _ = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*fourcc)

    video = cv2.VideoWriter(
        str(out),
        fourcc,
        fps,
        (width, height),
    )
    for img in frames:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(img)

    video.release()


class ExportVideo(RL_EvalCallback):
    training_loop: PPO
    env: BaseHumanoid

    def __init__(self, config, training_loop: PPO):
        super().__init__(config, training_loop)
        self.record_dir = Path(config.record_dir)
        self.record_dir.mkdir(exist_ok=True, parents=True)

    def on_pre_evaluate_policy(self):
        # Doing this in two lines because of type annotation issues.
        env: BaseHumanoid = self.training_loop.env
        self.env = env

    def on_post_evaluate_policy(self):
        self.write_recordings()

    def write_recordings(self):
        for idx in range(len(self.env.cameras)):
            if not self.config.only_record_viewer:
                frames = self.env.frames[idx]

                # cpu_frames = torch.stack(frames).cpu().numpy()

                # slower but more memory efficient
                cpu_frames = [f for f in frames]

                save_dir = self.record_dir / f"{(idx + self.config.index_offset):03d}"
                save_dir.mkdir(exist_ok=True, parents=True)

                if self.config.store_raw:
                    cat_frames = np.stack(cpu_frames)
                    with open(save_dir / "raw_frames.npy", "wb") as f:
                        np.save(f, cat_frames)

                else:
                    write_frames_to_video(
                        cpu_frames,
                        save_dir / f"video.{self.config.suffix}",
                        self.config.record_fps,
                    )

            self.env.frames[idx] = []

        if self.config.record_viewer:
            # load all saved frames in self.config.viewer_record_dir and combine into a video
            images = [
                img
                for img in os.listdir(self.config.viewer_record_dir)
                if img.endswith(".png")
            ]
            images.sort()
            sample_frame = cv2.imread(
                os.path.join(self.config.viewer_record_dir, images[0])
            )
            height, width, layers = sample_frame.shape

            fourcc = SUFFIX_TO_FOURCC[self.config.suffix]
            fourcc = cv2.VideoWriter_fourcc(*fourcc)
            video = cv2.VideoWriter(
                str(self.record_dir / f"viewer_video.{self.config.suffix}"),
                fourcc,
                30,
                (width, height),
            )

            for image in images:
                video.write(
                    cv2.imread(os.path.join(self.config.viewer_record_dir, image))
                )

            cv2.destroyAllWindows()
            video.release()

            # delete all images
            for image in images:
                os.remove(os.path.join(self.config.viewer_record_dir, image))
