# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import carb
import numpy as np
from pxr import Gf, Sdf
import omni.replicator.core as rep


class PerspectiveViewer(object):
    def __init__(self):
        self.viewport_api = None
        self.get_viewport_api()
        _ = rep.create.render_product(
            "/OmniverseKit_Persp", resolution=(500, 500)
        )  # Lower resolution

        # Disable advanced rendering features
        self.disable_advanced_rendering()

    def disable_advanced_rendering(self):
        stage = self.viewport_api.stage
        render_settings_path = "/Render/RenderProduct/RenderSettings"

        # Create or get the RenderSettings prim
        render_settings = stage.GetPrimAtPath(render_settings_path)
        if not render_settings.IsValid():
            render_settings = stage.DefinePrim(render_settings_path, "RenderSettings")

        # Disable ray tracing
        render_settings.CreateAttribute(
            "rtx:raytracing:enabled", Sdf.ValueTypeNames.Bool
        ).Set(False)

        # Disable Global Illumination
        render_settings.CreateAttribute(
            "rtx:pathtracing:gi:enabled", Sdf.ValueTypeNames.Bool
        ).Set(False)

        # Disable Ambient Occlusion
        render_settings.CreateAttribute(
            "rtx:ambientOcclusion:enabled", Sdf.ValueTypeNames.Bool
        ).Set(False)

        # Disable Depth of Field
        render_settings.CreateAttribute("rtx:dof:enabled", Sdf.ValueTypeNames.Bool).Set(
            False
        )

        # Optionally, you can also reduce other quality settings
        render_settings.CreateAttribute(
            "rtx:pathtracing:maxBounces", Sdf.ValueTypeNames.Int
        ).Set(1)
        render_settings.CreateAttribute(
            "rtx:pathtracing:maxSamples", Sdf.ValueTypeNames.Int
        ).Set(16)

        # Apply the changes
        stage.SetEditTarget(stage.GetSessionLayer())

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
