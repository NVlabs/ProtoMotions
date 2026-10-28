# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The control component contract.

A control component owns a task's episode state (a heading schedule, a path,
a goal prompt, the reference clock's consumers) and publishes it into the
context the observation, reward and termination kernels read. It is one
class, constructed as ``__init__(config, inputs: TaskInputs)``, and reads
``self.inputs`` and nothing else (``task_inputs.py`` says what that carries;
``ControlManager.check_holds_no_handles`` refuses an env, simulator, motion
library or manager held as an attribute).

Per-env state is allocated through ``self.state``, an
:class:`~protomotions.envs.base_env.episode_tensors.EpisodeTensors` (a helper
object gets a child registry, ``self.state.child(name)``): the registry resets,
snapshots and restores it. ``__setattr__`` guards the control's own attributes:
on any assignment once the registry exists, a bare tensor with a ``num_envs``
first dimension that the registry does not own raises. The guard sees only the
control's own tensor attributes, not tensors inside helper objects or
containers, nor Python scalars; those go through a child registry or a
hand-written ``save_state`` / ``load_state`` (``ExternalKinematicControl``'s
``_next_pose``), and ``tests/test_control_contract.py`` walks every in-tree
control and its helpers two levels deep for a bare per-env tensor.

Lifecycle, driven by the env and never by the control itself:

- ``reset_rows(env_ids, state)``: the orchestrator's reset; the default clears
  the registry's rows, a control then does its own sampling. A control never
  calls it on itself (a static test over ``envs/control`` pins that).
- ``advance(state)``: once per step after physics and the history record, on
  the robot state the env read; returns the logging dict for ``extras``. A
  schedule change inside an episode goes through ``resample(env_ids, state)``,
  so "who called me" is a question that cannot arise.
- ``publish(env_ids)``: pure; ``{path: value}`` for the names declared under
  the component's namespaces in ``context_views.py``.
- ``terminations(config, robot)``: a classmethod returning the episode-ending
  checks as bound ``TerminationRule`` entries the termination manager runs
  from the context (a boolean kernel, or a cut over a scalar kernel written
  beside the control; ``truncation=True`` ends the episode without failing).
- ``save_state`` / ``load_state``: the registry's.
- ``create_visualization_markers(headless)`` / ``get_markers_state(state)`` /
  ``close()``: the viewer's.

Every random draw passes ``generator=self.inputs.generator``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple, TYPE_CHECKING

import torch
from torch import Tensor

from protomotions.envs.base_env.episode_tensors import EpisodeTensors
from protomotions.envs.control.task_inputs import TaskInputs

if TYPE_CHECKING:
    from protomotions.envs.terminations.rule import TerminationRule
    from protomotions.simulator.base_simulator.config import (
        MarkerState,
        VisualizationMarkerConfig,
    )
    from protomotions.simulator.base_simulator.simulator_state import RobotState


@dataclass
class ControlComponentConfig:
    """Base configuration for control components.

    Components are enabled by being present in the control_components dict;
    to disable one, remove it from the dict.
    """


class ControlComponent(ABC):
    """One task's episode state and its context names; see the module docstring."""

    # The context namespaces this component's ``publish`` fills, as declared
    # with ``control_namespace(...)`` in ``context_views.py``; empty for a
    # component that publishes nothing. A binding to one of them is refused,
    # before anything is built, when no configured component lists it here.
    namespaces: Tuple[str, ...] = ()

    @classmethod
    def provides(cls, config: ControlComponentConfig, tag: str) -> bool:
        """Whether ``config`` turns on ``tag``, a requirement named on one of
        this component's context fields that is not a sensor, robot or
        history fact (see ``context_requirements.REQUIREMENTS``). By default
        the tag is the name of a config field and "on" means truthy, so
        ``requires="style"`` on a field reads ``config.style``. Override only
        when the switch is not a plain field. Reads config fields only: no
        data, no simulator, no random number.
        """
        return bool(getattr(config, tag, False))

    @classmethod
    def check_bindings(
        cls, config: ControlComponentConfig, robot_config, bound_paths: FrozenSet[str]
    ) -> None:
        """Refuse bindings this configuration cannot serve.

        ``bound_paths`` is every context path the observation, reward and
        termination components bind (:func:`bound_paths`). The build calls
        this for every configured component after the name checks, so a
        component whose published layout depends on its config (the masked
        mimic goal canvas) can refuse a binding to a stream it does not carry,
        or a switch nothing reads, with the reason. Reads config fields and
        the robot config only; the default refuses nothing.
        """

    @classmethod
    def terminations(
        cls, config: ControlComponentConfig, robot_config
    ) -> Dict[str, "TerminationRule"]:
        """The episode-ending checks this configuration owns, by name.

        Each entry is a ``TerminationRule`` over a kernel bound with
        ``MdpComponent.from_kernel`` (``envs/terminations/rule.py``): no
        threshold for a boolean kernel, a cut for a scalar one, and
        ``truncation=True`` for a check that ends the episode without counting
        as a failure. The termination manager validates and runs them from
        the context under ``<control>/<check>``, with the recipe's own
        terminations. Reads config fields and the robot config only. The
        default owns no check.
        """
        return {}

    def __init__(self, config: ControlComponentConfig, inputs: TaskInputs):
        self.config = config
        self.inputs = inputs
        self.state = EpisodeTensors(inputs.num_envs, inputs.device)

    def __setattr__(self, name: str, value: Any) -> None:
        # Once the registry exists, a per-env tensor must come from it: a bare
        # attribute would escape row reset, snapshot and restore. This sees the
        # control's own attributes only (the module docstring says the scope).
        state = self.__dict__.get("state")
        if (
            state is not None
            and torch.is_tensor(value)
            and value.ndim >= 1
            and value.shape[0] == self.inputs.num_envs
            and not state.owns(value)
        ):
            raise TypeError(
                f"{type(self).__name__}.{name} is a bare tensor with a num_envs first "
                "dimension; allocate it with self.state.add(...) (per-env state) or "
                "register it with self.state.constant(...) (a constant)"
            )
        super().__setattr__(name, value)

    def reset_rows(self, env_ids: Tensor, state: "RobotState") -> None:
        """Reset the task state of ``env_ids``, given their post-reset robot state.

        Called by the orchestrator only (``BaseEnv.reset``'s controls phase,
        after the physical reset has been read). The default puts the
        registry's rows back to their reset values; a control that samples
        new task parameters does so after calling it (or instead of it, when
        a slot deliberately carries over the previous episode's value).
        """
        self.state.reset_rows(env_ids)

    def resample(self, env_ids: Tensor, state: "RobotState") -> None:
        """Change the schedule of ``env_ids`` inside their episodes (a new
        heading, a new goal); called from ``advance`` when a change is due.
        The default has no schedule."""

    @abstractmethod
    def advance(self, state: "RobotState") -> Dict[str, Tensor]:
        """Update the task state once per env step, on the robot state the env
        just read (after physics, after the history record, before the context
        is built). Returns the logging dict merged into ``extras``; a control
        with nothing to log returns ``{}``.
        """

    @abstractmethod
    def publish(self, env_ids: Optional[Tensor]) -> Dict[str, Any]:
        """Return the context names this component publishes.

        The keys are the paths declared under the component's namespace in
        ``context_views.py`` (``"mimic.future_pos"``, ``"steering.tar_dir"``);
        the values are the task tensors, sliced to ``env_ids`` when given
        (None means every env). The env merges every component's dict into
        ``env.context``; a name published twice, or one the vocabulary does
        not declare, fails the build. A component with no task data returns
        ``{}``. Pure: no state changes, no random draws, no robot state read.

        Example:
            >>> def publish(self, env_ids):
            ...     rows = slice(None) if env_ids is None else env_ids
            ...     return {"target.tar_pos": self.tar_pos[rows]}
        """

    def save_state(self) -> Dict[str, Any]:
        """The registry's snapshot (owned clones), for the env's snapshot walk."""
        return self.state.save_state()

    def load_state(self, snapshot: Dict[str, Any]) -> None:
        """Write a ``save_state`` result back in place."""
        self.state.load_state(snapshot)

    def create_visualization_markers(
        self, headless: bool
    ) -> Dict[str, "VisualizationMarkerConfig"]:
        """Marker configurations for the viewer, allocated once at build;
        empty when headless. The default has no markers."""
        return {}

    def get_markers_state(self, state: "RobotState") -> Dict[str, "MarkerState"]:
        """Where every marker is this frame, from the robot state the viewer
        read; empty when headless. The default has no markers."""
        return {}

    def close(self) -> None:
        """Release resources owned by this component (UI key scopes)."""


def bound_paths(*component_tables: Mapping[str, Any]) -> FrozenSet[str]:
    """Every context path the components of ``component_tables`` bind (the
    argument ``ControlComponent.check_bindings`` receives)."""
    return frozenset(
        field_path.path
        for table in component_tables
        for component in (table or {}).values()
        for field_path in component.dynamic_vars.values()
    )


__all__ = ["ControlComponent", "ControlComponentConfig", "bound_paths"]
