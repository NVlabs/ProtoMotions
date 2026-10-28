# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The control components, constructed from one TaskInputs and driven as one.

``ControlManager`` builds every configured component as ``cls(config, inputs)``
and fans the env's calls out to them: ``reset_rows`` at a reset, ``advance``
once per step (merging their logging, refusing a key two components both
log), ``terminations`` for the termination manager's table (keyed
``<control>/<check>``), the snapshot walk, the viewer's markers and
``close``. ``check_holds_no_handles`` is the build-time walk that refuses a
component holding the env, the simulator, the motion library or the motion
manager: a control reads only its TaskInputs.

``TeleportQueue`` is the animation players' declared write channel: a
component queues a pose for some envs during ``advance`` and the env applies
the queue to the simulator once every component has advanced
(``ControlsState.advance``), so no component holds the simulator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple, TYPE_CHECKING

from torch import Tensor

from protomotions.envs.context_requirements import control_terminations
from protomotions.envs.control.task_inputs import TaskInputs
from protomotions.utils.hydra_replacement import get_class

if TYPE_CHECKING:
    from protomotions.envs.control.base import ControlComponent
    from protomotions.envs.terminations.rule import TerminationRule
    from protomotions.simulator.base_simulator.config import (
        MarkerState,
        VisualizationMarkerConfig,
    )
    from protomotions.simulator.base_simulator.simulator_state import (
        ObjectState,
        ResetState,
        RobotState,
    )


class TeleportQueue:
    """Poses the animation players ask the env to write, applied in order."""

    def __init__(self):
        self._requests: List[Tuple[Tensor, "ResetState", Optional["ObjectState"]]] = []

    def request(
        self, env_ids: Tensor, robot: "ResetState", objects: Optional["ObjectState"]
    ) -> None:
        """Queue a pose write for ``env_ids`` (``TaskInputs.teleport``)."""
        self._requests.append((env_ids, robot, objects))

    def apply(self, simulator) -> None:
        """Write every queued pose to the simulator, in request order, then clear."""
        for env_ids, robot, objects in self._requests:
            simulator.reset_envs(robot, objects, env_ids)
        self._requests.clear()


class ControlManager:
    """The configured control components and the calls the env fans out to them."""

    def __init__(
        self,
        configs: Mapping[str, Any],
        inputs: TaskInputs,
        teleports: TeleportQueue,
    ):
        self.inputs = inputs
        self.teleports = teleports
        # Components are enabled by being present in the dict.
        self.components: Dict[str, "ControlComponent"] = {}
        for name, config in configs.items():
            self.components[name] = get_class(config._target_)(config, inputs)

    def check_holds_no_handles(self) -> None:
        """Refuse a component that holds the env, the simulator, the motion
        library or the motion manager (one level of ``vars``, ``inputs`` aside)."""
        # Imported here: the env imports the build, which imports this module.
        from protomotions.components.motion_lib import MotionLib
        from protomotions.envs.base_env.env import BaseEnv
        from protomotions.envs.motion_manager.motion_manager import MotionManager
        from protomotions.simulator.base_simulator.simulator import Simulator

        handles = (BaseEnv, Simulator, MotionLib, MotionManager)
        for component in self.components.values():
            for attribute, value in vars(component).items():
                if attribute == "inputs" or not isinstance(value, handles):
                    continue
                held = next(cls for cls in handles if isinstance(value, cls))
                raise TypeError(
                    f"{type(component).__name__}.{attribute} holds a {held.__name__}: "
                    "a control reads only its TaskInputs"
                )

    def terminations(self) -> Dict[str, "TerminationRule"]:
        """Every component's checks under ``<control>/<check>``
        (``context_requirements.control_terminations`` over their configs)."""
        return control_terminations(
            {
                name: (type(component), component.config)
                for name, component in self.components.items()
            },
            self.inputs.robot,
        )

    def advance(self, state: "RobotState") -> Dict[str, Tensor]:
        """Advance every component on the robot state the env read; one merged
        logging dict, a key logged twice refused by name."""
        logging: Dict[str, Tensor] = {}
        logged_by: Dict[str, str] = {}
        for name, component in self.components.items():
            logged = component.advance(state)
            if not isinstance(logged, dict):
                raise TypeError(
                    f"control component '{name}'.advance returned "
                    f"{type(logged).__name__}, not a dict of logging tensors "
                    "(return {} when there is nothing to log)"
                )
            for key, value in logged.items():
                if key in logging:
                    raise ValueError(
                        f"control components '{logged_by[key]}' and '{name}' both "
                        f"log '{key}'"
                    )
                logging[key] = value
                logged_by[key] = name
        return logging

    def reset_rows(self, env_ids: Tensor, state: "RobotState") -> None:
        """Reset every component's rows of ``env_ids`` (the env's reset)."""
        for component in self.components.values():
            component.reset_rows(env_ids, state)

    # The components publish into the context one by one: ``BaseEnv._publish``
    # walks ``components`` and calls each ``publish(env_ids)``, so an error
    # can name the component that published a value.

    def save_state(self) -> Dict[str, Any]:
        """Every component's registry snapshot, by component name."""
        return {
            name: component.save_state() for name, component in self.components.items()
        }

    def load_state(self, snapshot: Dict[str, Any]) -> None:
        for name, component in self.components.items():
            component.load_state(snapshot[name])

    def create_visualization_markers(
        self, headless: bool
    ) -> Dict[str, "VisualizationMarkerConfig"]:
        """Every component's marker configurations; empty when headless."""
        if headless:
            return {}
        markers: Dict[str, "VisualizationMarkerConfig"] = {}
        for component in self.components.values():
            markers.update(component.create_visualization_markers(headless))
        return markers

    def get_markers_state(self, state: "RobotState") -> Dict[str, "MarkerState"]:
        """Every component's marker states this frame, from one robot state read."""
        markers_state: Dict[str, "MarkerState"] = {}
        for component in self.components.values():
            markers_state.update(component.get_markers_state(state))
        return markers_state

    def close(self) -> None:
        for component in self.components.values():
            component.close()


__all__ = ["ControlManager", "TeleportQueue"]
