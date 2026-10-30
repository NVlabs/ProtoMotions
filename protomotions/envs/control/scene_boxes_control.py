# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The placeholder scene-box stream: the names a scene-conditioned policy binds.

A policy conditioned on scene boxes (the Metatron G1 release) binds the eight
``scene_boxes.*`` names (``context_views.SceneBoxesContext``). The environment
refuses a bound name nobody publishes, and one saved configuration has to both
run in the simulator and export, so this component publishes the names in the
simulator: a fixed-capacity, all-padding scene (every box invalid, every object
hidden, identity object poses, zero pose bias). Nothing here reads the scene
library; scene-box training is not supported, and a host supplies real values
at deployment (the exporter labels them with the ``scene`` provider,
``deployment/contract.py``). A scene library with objects is refused at
construction: the policy would be told there are no boxes while objects stand
in the simulator. A later component that fills the same names from the scene
library replaces ``publish`` and that refusal, and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.base import ControlComponent, ControlComponentConfig
from protomotions.envs.control.task_inputs import TaskInputs
from protomotions.simulator.base_simulator.simulator_state import RobotState


@dataclass
class SceneBoxesControlConfig(ControlComponentConfig):
    """The capacity of the scene-box stream: it fixes every published shape."""

    _target_: str = "protomotions.envs.control.scene_boxes_control.SceneBoxesControl"
    max_objects: int = 1
    max_boxes_per_object: int = 2

    def __post_init__(self) -> None:
        if self.max_objects < 1 or self.max_boxes_per_object < 1:
            raise ValueError(
                "SceneBoxesControlConfig needs at least one object and one box per "
                f"object, got max_objects={self.max_objects}, "
                f"max_boxes_per_object={self.max_boxes_per_object}"
            )


class SceneBoxesControl(ControlComponent):
    """Publishes ``scene_boxes`` as an all-padding scene; see the module docstring."""

    config: SceneBoxesControlConfig
    namespaces = ("scene_boxes",)

    def __init__(self, config: SceneBoxesControlConfig, inputs: TaskInputs):
        super().__init__(config, inputs)
        if inputs.reference.has_scene_objects:
            raise ValueError(
                "SceneBoxesControl publishes a scene with no boxes, but the scene "
                "library has objects: the policy would be told there are none. "
                "Scene-box training is not supported; drop the scene library or "
                "this control"
            )

    def advance(self, state: RobotState) -> Dict[str, Tensor]:
        return {}

    def publish(self, env_ids: Optional[Tensor]) -> Dict[str, Any]:
        rows = self.inputs.num_envs if env_ids is None else int(len(env_ids))
        device = self.inputs.device
        objects = self.config.max_objects
        boxes = self.config.max_boxes_per_object
        object_rot = torch.zeros(rows, objects, 4, device=device)
        object_rot[..., 3] = 1.0  # xyzw identity
        names = EnvContext.scene_boxes
        return {
            names.centers.path: torch.zeros(rows, objects, boxes, 3, device=device),
            names.extents.path: torch.zeros(rows, objects, boxes, 3, device=device),
            names.valid.path: torch.zeros(
                rows, objects, boxes, dtype=torch.bool, device=device
            ),
            names.visible.path: torch.zeros(
                rows, objects, 1, dtype=torch.bool, device=device
            ),
            names.object_pos.path: torch.zeros(rows, objects, 3, device=device),
            names.object_rot.path: object_rot,
            names.position_offset.path: torch.zeros(rows, 3, device=device),
            names.yaw_offset.path: torch.zeros(rows, device=device),
        }


__all__ = ["SceneBoxesControlConfig", "SceneBoxesControl"]
