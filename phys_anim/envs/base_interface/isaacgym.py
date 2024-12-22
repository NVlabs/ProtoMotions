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

import sys
from typing import TYPE_CHECKING

from isaacgym import gymapi  # type: ignore[misc]
import torch

from phys_anim.envs.base_interface.common import BaseInterface

if TYPE_CHECKING:
    # Import IsaacGym Humanoid for autocompletion. Not imported during runtime.
    from phys_anim.envs.humanoid.isaacgym import Humanoid
else:
    Humanoid = object


# Base class for RL tasks
class GymBaseInterface(BaseInterface, Humanoid):  # type: ignore[misc]
    def __init__(
        self,
        config,
        device: torch.device,
    ):
        super().__init__(config, device)
        # double check!
        self.graphics_device_id = self.device.index
        if self.headless is True:
            self.graphics_device_id = -1

        self.gym = gymapi.acquire_gym()

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if not self.headless:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_U, "update_inference_parameters"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_J, "apply_force"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_L, "toggle_video_record"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_SEMICOLON, "cancel_video_record"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_R, "reset_envs"
            )

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
                cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.init_done = True

    def get_obs_size(self):
        raise NotImplementedError

    # set gravity based on up axis and return axis index
    def set_sim_params_up_axis(self, sim_params, axis):
        if axis == "z":
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity.x = 0
            sim_params.gravity.y = 0
            sim_params.gravity.z = -9.81
            return 2
        return 1

    def create_sim(self):
        sim = self.gym.create_sim(
            self.device.index,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        if sim is None:
            print("*** Failed to create sim")
            quit()

        self.sim = sim

    def step(self, actions):
        # apply actions
        self.pre_physics_step(actions)

        # step physics and render each frame
        self.physics_step()

        # TODO: to fix!
        if self.device.type == "cpu":
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        self.render()

        return (
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def render(self):
        if not self.headless:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "update_inference_parameters" and evt.value > 0:
                    self.update_inference_parameters()
                elif evt.action == "apply_force" and evt.value > 0:
                    self.apply_sideways_force_to_feet()
                elif evt.action == "toggle_video_record" and evt.value > 0:
                    self.toggle_video_record()
                elif evt.action == "cancel_video_record" and evt.value > 0:
                    self.cancel_video_record()
                elif evt.action == "reset_envs" and evt.value > 0:
                    self.force_reset()

            # fetch results
            if self.device.type != "cpu":
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
            else:
                self.gym.poll_viewer_events(self.viewer)

        super().render()

    def write_viewport_to_file(self, file_name):
        self.gym.write_viewer_image_to_file(
            self.viewer,
            file_name,
        )

    def simulate(self):
        self.gym.simulate(self.sim)
        if self.device == "cpu":
            self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)

    def close(self):
        if self.viewer:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
