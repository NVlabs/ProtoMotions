# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The env's visualization markers: the terrain height-map markers and the
control components' markers, merged.

Two calls, both empty in headless mode: ``configs`` is what the simulator
allocates its marker set from (``build_env``'s markers phase, before the
simulator's GPU allocation), ``state`` is the per-frame callback the physics
phase of ``BaseEnv.step`` hands to ``simulator.step``. The terrain markers
exist only with ``config.show_terrain_markers``; the control components'
come from the control manager, from one robot-state read per frame. Holds
the pieces it reads, not the env.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch

from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    MarkerState,
    VisualizationMarkerConfig,
)

TERRAIN_MARKERS = "terrain_markers"


class Markers:
    """Marker configs and per-frame marker states, for a viewer."""

    def __init__(
        self,
        control_manager,
        terrain,
        get_root_state: Callable[[], object],
        get_robot_state: Callable[[], object],
        *,
        show_terrain_markers: bool,
        headless: bool,
        num_envs: int,
        device: torch.device,
    ):
        self.control_manager = control_manager
        self.terrain = terrain
        self.get_root_state = get_root_state
        self.get_robot_state = get_robot_state
        self.show_terrain_markers = show_terrain_markers
        self.headless = headless
        self.num_envs = num_envs
        self.device = device

    def configs(self) -> Dict[str, VisualizationMarkerConfig]:
        """The marker set the simulator allocates; empty when headless."""
        if self.headless:
            return {}
        markers: Dict[str, VisualizationMarkerConfig] = {}
        if self.show_terrain_markers:
            markers[TERRAIN_MARKERS] = VisualizationMarkerConfig(
                type="sphere",
                color=(0.008, 0.345, 0.224),
                markers=[
                    MarkerConfig(size="small")
                    for _ in range(self.terrain.num_height_points)
                ],
            )
        markers.update(self.control_manager.create_visualization_markers(self.headless))
        return markers

    def state(self) -> Dict[str, MarkerState]:
        """Where every marker is this frame; empty when headless."""
        if self.headless:
            return {}
        state: Dict[str, MarkerState] = {}
        if self.show_terrain_markers:
            height_maps = self.terrain.get_height_maps(
                self.get_root_state(), None, return_all_dims=True
            ).view(self.num_envs, -1, 3)
            state[TERRAIN_MARKERS] = MarkerState(
                translation=height_maps,
                orientation=torch.zeros(
                    self.num_envs, height_maps.shape[1], 4, device=self.device
                ),
            )
        # One robot-state read per frame, handed to every control component.
        state.update(self.control_manager.get_markers_state(self.get_robot_state()))
        return state


__all__ = ["Markers", "TERRAIN_MARKERS"]
