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

import carb
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom


class PerspectiveViewer(object):
    def __init__(self):
        self.viewport_api = None
        self.get_viewport_api()

    def get_viewport_api(self):
        if self.viewport_api is None:
            try:
                from omni.kit.viewport.utility import get_active_viewport

                self.viewport_api = get_active_viewport()
            except ImportError:
                carb.log_warn(
                    "omni.kit.viewport.utility needs to be enabled before using this function"
                )

            if self.viewport_api is None:
                carb.log_warn("could not get active viewport, cannot set camera view")

    def get_camera_state(self):
        self.get_viewport_api()

        from omni.kit.viewport.utility.camera_state import ViewportCameraState

        prim = self.viewport_api.stage.GetPrimAtPath("/OmniverseKit_Persp")

        coi_prop = prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            prim.CreateAttribute(
                "omni:kit:centerOfInterest",
                Sdf.ValueTypeNames.Vector3d,
                True,
                Sdf.VariabilityUniform,
            ).Set(Gf.Vec3d(0, 0, -10))
        camera_state = ViewportCameraState("/OmniverseKit_Persp", self.viewport_api)
        camera_position = camera_state.position_world
        return camera_position[0], camera_position[1], camera_position[2]

    def set_camera_view(self, eye: np.array, target: np.array):
        self.get_viewport_api()

        from omni.kit.viewport.utility.camera_state import ViewportCameraState

        camera_position = np.asarray(eye, dtype=np.double)
        camera_target = np.asarray(target, dtype=np.double)
        prim = self.viewport_api.stage.GetPrimAtPath("/OmniverseKit_Persp")

        coi_prop = prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            prim.CreateAttribute(
                "omni:kit:centerOfInterest",
                Sdf.ValueTypeNames.Vector3d,
                True,
                Sdf.VariabilityUniform,
            ).Set(Gf.Vec3d(0, 0, -10))
        camera_state = ViewportCameraState("/OmniverseKit_Persp", self.viewport_api)
        camera_state.set_position_world(
            Gf.Vec3d(camera_position[0], camera_position[1], camera_position[2]), True
        )
        camera_state.set_target_world(
            Gf.Vec3d(camera_target[0], camera_target[1], camera_target[2]), True
        )
