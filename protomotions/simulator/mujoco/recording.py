# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offscreen capture of MuJoCo's live camera and debug geometry.

Callers hold the passive viewer lock while reading its state and rendering.
Clip lifetime and video encoding remain in the shared recording mixin.
"""

from __future__ import annotations

import mujoco
import numpy as np


def recording_dimensions(viewer) -> tuple[int, int]:
    """Return even viewport dimensions (width, height), or a 640×480 fallback."""
    viewport = getattr(viewer, "viewport", None)
    width, height = 640, 480
    if viewport is not None and min(viewport.width, viewport.height) > 0:
        width, height = viewport.width, viewport.height
    # H.264's usual yuv420p pixel format requires even dimensions.
    return max(2, int(width) // 2 * 2), max(2, int(height) // 2 * 2)


def render_viewport(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer,
    width: int,
    height: int,
) -> np.ndarray:
    """Render RGB pixels with the live camera, options, flags, and markers."""
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    # update_scene rebuilds model geometry only; user_scn holds goal markers
    # and arrows that must be appended to the offscreen scene separately.
    debug_scene = viewer.user_scn if viewer is not None else None
    renderer = mujoco.Renderer(
        model,
        height=height,
        width=width,
        max_geom=max(10000, model.ngeom)
        + (debug_scene.maxgeom if debug_scene is not None else 0),
    )
    try:
        if viewer is not None:
            renderer.update_scene(data, camera=viewer.cam, scene_option=viewer.opt)
            scene = renderer.scene
            scene.flags[:] = debug_scene.flags
            for i in range(debug_scene.ngeom):
                source = debug_scene.geoms[i]
                dest = scene.geoms[scene.ngeom]
                for field in ("size", "pos", "mat", "rgba"):
                    getattr(dest, field)[:] = getattr(source, field)
                # Preserve material, label and segmentation metadata too.
                for field in (
                    "type",
                    "dataid",
                    "objtype",
                    "objid",
                    "category",
                    "matid",
                    "texcoord",
                    "segid",
                    "emission",
                    "specular",
                    "shininess",
                    "reflectance",
                    "camdist",
                    "modelrbound",
                    "transparent",
                    "label",
                ):
                    setattr(dest, field, getattr(source, field))
                scene.ngeom += 1
        else:
            renderer.update_scene(data)
        return renderer.render()
    finally:
        renderer.close()
