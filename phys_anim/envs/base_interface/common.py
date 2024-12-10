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

import torch
import os
from collections import deque
import os
from datetime import datetime


class BaseInterface(object):
    def __init__(self, config, device: torch.device, *args, **kwargs):
        self.w_last = True  # We convert everything to w_last
        self.config = config
        self.device = device
        self.headless = config.headless

        self.num_envs = config.num_envs

        if self.config.sync_motion:
            control_freq_inv = self.config.simulator.sim.control_freq_inv
            self.config.simulator.sim.control_freq_inv = 1
            self.sync_motion_dt = control_freq_inv / config.simulator.sim.fps
            print("HACK SLOW DOWN")
            self.config.robot.control.control_type = "T"

        self.isaac_pd = self.config.robot.control.control_type == "isaac_pd"
        self.control_freq_inv = self.config.simulator.sim.control_freq_inv

        self.dt: float = (
            self.config.simulator.sim.control_freq_inv
            * 1.0
            / self.config.simulator.sim.fps
        )
        
        self.user_is_recording, self.user_recording_state_change = False, False
        self.user_recording_video_queue_size = 100000
        self.delete_user_viewer_recordings = False
        rendering_out = os.path.join("output", "renderings")
        os.makedirs(rendering_out, exist_ok=True)
        self.user_recording_video_path = os.path.join(
            rendering_out, f"{self.config.experiment_name}-%s"
        )

    def get_obs_size(self):
        raise NotImplementedError

    def on_environment_ready(self):
        pass

    def step(self, actions):
        raise NotImplementedError

    def pre_physics_step(self, actions):
        raise NotImplementedError

    def reset(self, env_ids=None):
        raise NotImplementedError

    def physics_step(self):
        if self.isaac_pd:
            self.apply_pd_control()
        for i in range(self.control_freq_inv):
            if not self.isaac_pd:
                self.apply_motor_forces()
            self.simulate()

    def force_reset(self):
        print("Resetting envs")

    def simulate(self):
        raise NotImplementedError

    def post_physics_step(self):
        raise NotImplementedError

    def on_epoch_end(self, current_epoch: int):
        pass

    def close(self):
        raise NotImplementedError

    def render(self):
        if not self.headless:
            if self.user_recording_state_change:
                if self.user_is_recording:
                    self.user_recording_video_queue = deque(
                        maxlen=self.user_recording_video_queue_size
                    )

                    curr_date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                    self.curr_user_recording_name = (
                        self.user_recording_video_path % curr_date_time
                    )
                    self.user_recording_frame = 0
                    if not os.path.exists(self.curr_user_recording_name):
                        os.makedirs(self.curr_user_recording_name)

                    print(
                        f"Started to record data into folder {self.curr_user_recording_name}"
                    )
                if not self.user_is_recording:
                    import cv2
                    images = [
                        img
                        for img in os.listdir(self.curr_user_recording_name)
                        if img.endswith(".png")
                    ]
                    images.sort()
                    sample_frame = cv2.imread(
                        os.path.join(self.curr_user_recording_name, images[0])
                    )
                    height, width, layers = sample_frame.shape

                    fourcc = "MP4V"
                    fourcc = cv2.VideoWriter_fourcc(*fourcc)
                    video = cv2.VideoWriter(
                        str(self.curr_user_recording_name) + ".mp4",
                        fourcc,
                        30,
                        (width, height),
                    )

                    for image in images:
                        video.write(
                            cv2.imread(
                                os.path.join(self.curr_user_recording_name, image)
                            )
                        )

                    cv2.destroyAllWindows()
                    video.release()

                    self.delete_user_viewer_recordings = True

                    print(
                        f"============ Video finished writing {self.curr_user_recording_name}.mp4 ============"
                    )
                else:
                    print("============ Writing video ============")
                self.user_recording_state_change = False

            if self.user_is_recording:
                file_name = self.curr_user_recording_name + "/%04d.png" % self.user_recording_frame
                self.write_viewport_to_file(
                    file_name
                )
                self.user_recording_frame += 1

            if self.delete_user_viewer_recordings:
                images = [
                    img
                    for img in os.listdir(self.curr_user_recording_name)
                    if img.endswith(".png")
                ]
                # delete all images
                for image in images:
                    os.remove(os.path.join(self.curr_user_recording_name, image))
                os.removedirs(self.curr_user_recording_name)

            self.delete_user_viewer_recordings = False

    def toggle_video_record(self):
        self.user_is_recording = not self.user_is_recording
        self.user_recording_state_change = True
        
    def cancel_video_record(self):
        self.user_is_recording = False
        self.user_recording_state_change = False
        self.delete_user_viewer_recordings = True
