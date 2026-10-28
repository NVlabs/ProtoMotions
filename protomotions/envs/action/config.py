# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The typed action configuration.

An ``ActionConfig`` binds a pure action function to the parameters it reads,
checked against the function's signature at creation. It replaces the bare
``{"fn": ..., <param>: ...}`` dict: ``apply(action)`` runs the function,
``to(device)`` moves the tensor parameters, and ``tensors()`` / ``constants()``
split them for the ONNX exporter. The ``make_*_action_config`` helpers return
one. The module imports only torch, inspect, functools, dataclasses and the pure
action functions, so it can be imported at module level wherever a config is
declared without a cycle.
"""

import functools
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Union

import torch
from torch import Tensor

from protomotions.envs.action.action_functions import (
    ActionTransform,
    bm_pd_action,
    build_pd_action_offset_scale,
    normalized_pd_fixed_gains_action,
    passthrough_pd_action,
)


@dataclass
class ActionConfig:
    """A pure action function bound to the parameters it reads.

    ``fn`` is a plain module-level function taking ``action`` first; ``params``
    holds every other argument it needs (tensors and constants). The binding is
    checked against ``fn``'s signature at creation, so a typo or a missing
    parameter fails loudly here instead of at the first step.
    """

    fn: Callable[..., Dict[str, Tensor]]
    params: Dict[str, Union[Tensor, float, int, str, None]]

    def __post_init__(self) -> None:
        # A partial has no __name__; a C/TorchScript callable may have neither
        # name nor qualname; a method or nested function has a dotted qualname;
        # a lambda is named "<lambda>" at any scope. Each hides from a pickle
        # that stores the function by its import path, so refuse them all here.
        name = getattr(self.fn, "__name__", None)
        qualname = getattr(self.fn, "__qualname__", None)
        if (
            isinstance(self.fn, functools.partial)
            or name is None
            or qualname is None
            or qualname != name
            or name == "<lambda>"
        ):
            raise TypeError(
                "ActionConfig.fn must be a plain module-level function "
                "(no partial, no lambda, no method)"
            )
        if "action" in self.params:
            raise ValueError(
                "ActionConfig.params must not contain 'action'; apply() supplies it"
            )
        parameters = inspect.signature(self.fn).parameters
        for key in self.params:
            if key not in parameters:
                raise ValueError(
                    f"action function {self.fn.__name__} has no parameter {key!r}"
                )
        for name, parameter in parameters.items():
            if name == "action" or parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if parameter.default is inspect.Parameter.empty and name not in self.params:
                raise ValueError(
                    f"ActionConfig for {self.fn.__name__} is missing {name!r}"
                )

    def apply(self, action: Tensor) -> Dict[str, Tensor]:
        """Run the action function on ``action`` and the bound parameters."""
        return self.fn(action=action, **self.params)

    def to(self, device: torch.device) -> "ActionConfig":
        """A copy with every tensor parameter moved to ``device``."""
        moved = {
            key: (value.to(device) if isinstance(value, torch.Tensor) else value)
            for key, value in self.params.items()
        }
        return ActionConfig(fn=self.fn, params=moved)

    def tensors(self) -> Dict[str, Tensor]:
        """The tensor parameters (the exporter registers these as buffers)."""
        return {
            key: value
            for key, value in self.params.items()
            if isinstance(value, torch.Tensor)
        }

    def constants(self) -> Dict[str, Any]:
        """The non-tensor parameters (the exporter bakes these in as constants)."""
        return {
            key: value
            for key, value in self.params.items()
            if not isinstance(value, torch.Tensor)
        }

    @property
    def num_actions(self) -> int:
        """The action width, read off the first 1-D tensor parameter."""
        for value in self.params.values():
            if isinstance(value, torch.Tensor) and value.dim() == 1:
                return value.shape[0]
        raise ValueError(
            f"ActionConfig for {self.fn.__name__} has no 1-D tensor parameter to "
            f"take num_actions from"
        )

    def neutral_command(self) -> Tensor:
        """The reset action: zeros of width ``num_actions`` (today's reset fill)."""
        return torch.zeros(self.num_actions)


def make_pd_action_config(
    robot_config,
    action_transform: ActionTransform = "tanh",
    clamp_value: float = 1.0,
    action_scale: float = 1.0,
) -> ActionConfig:
    """Normalized PD control: bound actions scaled into the per-joint PD range.

    Args:
        robot_config: Robot configuration with kinematic_info and control fields.
        action_transform: How to bound actions - "tanh" (default), "clamp", or None.
        clamp_value: Max absolute action value for clamp mode.
        action_scale: Scale factor beyond the PD range so the motor does not
            lose strength near joint limits.
    """
    pd_action_offset, pd_action_scale = build_pd_action_offset_scale(
        robot_config.kinematic_info.hinge_axes_map,
        robot_config.kinematic_info.dof_limits_lower,
        robot_config.kinematic_info.dof_limits_upper,
        action_scale,
        torch.device("cpu"),
    )

    joint_names = robot_config.kinematic_info.dof_names
    control_info = robot_config.control.control_info
    stiffness = torch.tensor(
        [control_info[j].require("stiffness") for j in joint_names],
        dtype=torch.float32,
    )
    damping = torch.tensor(
        [control_info[j].require("damping") for j in joint_names],
        dtype=torch.float32,
    )

    return ActionConfig(
        fn=normalized_pd_fixed_gains_action,
        params={
            "pd_action_offset": pd_action_offset,
            "pd_action_scale": pd_action_scale,
            "stiffness": stiffness,
            "damping": damping,
            "action_transform": action_transform,
            "clamp_value": clamp_value,
        },
    )


def make_bm_pd_action_config(
    robot_config, effort_fraction: float = 0.25
) -> ActionConfig:
    """BeyondMimic-style PD scaling: ``effort_fraction * effort_limit / stiffness``.

    Unit policy output produces a proportional torque of ``effort_fraction`` of
    the effort limit at the default pose. The offset is ``default_dof_pos``.
    Gains and limits are read through ``ControlInfo.require`` so a missing one
    fails loudly instead of defaulting to a silent constant.
    """
    joint_names = robot_config.kinematic_info.dof_names
    control_info = robot_config.control.control_info
    stiffness = torch.tensor(
        [control_info[j].require("stiffness") for j in joint_names],
        dtype=torch.float32,
    )
    damping = torch.tensor(
        [control_info[j].require("damping") for j in joint_names],
        dtype=torch.float32,
    )
    effort_limit = torch.tensor(
        [control_info[j].require("effort_limit") for j in joint_names],
        dtype=torch.float32,
    )

    if torch.any(stiffness <= 0):
        raise ValueError("BM PD action stiffness must be positive for all joints")

    action_scale = effort_fraction * effort_limit / stiffness
    pd_action_offset = robot_config.default_dof_pos.clone()

    return ActionConfig(
        fn=bm_pd_action,
        params={
            "pd_action_offset": pd_action_offset,
            "action_scale": action_scale,
            "stiffness": stiffness,
            "damping": damping,
        },
    )


def make_passthrough_pd_action_config(robot_config) -> ActionConfig:
    """Passthrough PD: the policy output is the joint position target, no transform."""
    joint_names = robot_config.kinematic_info.dof_names
    control_info = robot_config.control.control_info
    stiffness = torch.tensor(
        [control_info[j].require("stiffness") for j in joint_names],
        dtype=torch.float32,
    )
    damping = torch.tensor(
        [control_info[j].require("damping") for j in joint_names],
        dtype=torch.float32,
    )

    return ActionConfig(
        fn=passthrough_pd_action,
        params={
            "stiffness": stiffness,
            "damping": damping,
        },
    )


__all__ = [
    "ActionConfig",
    "make_bm_pd_action_config",
    "make_passthrough_pd_action_config",
    "make_pd_action_config",
]
