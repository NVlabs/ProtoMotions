# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
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
"""MdpComponent: Binds pure tensor compute functions to context paths.

MdpComponent separates:
- compute_func: Pure tensor function (Level 2, exportable)
- dynamic_vars: Maps compute_func params to FieldPath context paths
- static_params: Constants not from context (scalars, fixed tensors, metadata)

Connecting new data to a compute function
==========================================

Every compute_func argument must come from exactly one of two places:

**dynamic_vars** — Runtime values that change every step.
    These MUST be ``FieldPath`` objects obtained from class-level access on
    ``EnvContext`` (e.g. ``EnvContext.current.dof_pos``).  At runtime,
    ``compute()`` resolves each path from the live context instance.  At ONNX
    export time, ``get_bindings_dict()`` turns them into named inputs.

    If your data is not yet on ``EnvContext``, add a new field:
    1. Add a ``FieldPath()`` descriptor to the appropriate view class in
       ``context_views.py`` (or create a new view class + ``NestedField``).
    2. Populate the field in the ``__init__`` of that view or in the
       controller's ``populate_context()``.
    3. Reference it via ``EnvContext.<view>.<field>`` in ``dynamic_vars``.

**static_params** — Values fixed for the lifetime of the environment.
    Scalars (``float``, ``int``, ``bool``, ``str``), fixed tensors (e.g.
    ``dof_limits_lower`` from robot config), and reward/termination metadata
    (``weight``, ``min_value``, etc.) all go here.  Fixed tensors are
    automatically moved to the runtime device on the first ``compute()`` call.

    Do NOT put tensors or other concrete values in ``dynamic_vars``.
    Doing so will raise an ``AttributeError`` because ``compute()`` expects
    every binding value to be a ``FieldPath``.

Metadata keys
-------------
Some ``static_params`` keys are *metadata* consumed by the reward/termination
combining logic, not by the compute_func itself.  These are automatically filtered
out before calling the compute_func.  Current metadata keys:

    weight, multiplicative, zero_during_grace_period,
    min_value, max_value, use_region_weights

Example
-------
::

    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent

    observation_components = {
        "max_coords_obs": MdpComponent(
            compute_func=compute_humanoid_max_coords_observations,
            dynamic_vars={
                "body_pos": EnvContext.current.rigid_body_pos,  # FieldPath
                "body_rot": EnvContext.current.rigid_body_rot,
            },
            static_params={"local_obs": True},
        ),
    }

    reward_components = {
        "limits_dof_pos": MdpComponent(
            compute_func=compute_soft_pos_limit_rew,
            dynamic_vars={
                "dof_pos": EnvContext.current.dof_pos,          # FieldPath (dynamic)
            },
            static_params={
                "weight": -100.0,                               # metadata
                "dof_limits_lower": robot_cfg.kinematic_info.dof_limits_lower,  # fixed tensor
                "dof_limits_upper": robot_cfg.kinematic_info.dof_limits_upper,  # fixed tensor
            },
        ),
    }
"""

from __future__ import annotations

from typing import Any, Callable, Dict, TYPE_CHECKING

from torch import Tensor

from protomotions.envs.context_paths import FieldPath, resolve_path

if TYPE_CHECKING:
    from protomotions.envs.context_views import EnvContext


# Keys in static_params that are reward/termination metadata, NOT compute_func arguments.
# These are consumed by combine_rewards / combine_terminations and must not
# be forwarded to the compute_func function.
_METADATA_KEYS = frozenset({
    "weight", "multiplicative", "zero_during_grace_period",
    "min_value", "max_value", "use_region_weights",
    "threshold", "fail_above",
})


class MdpComponent:
    """A compute function with explicit context bindings.
    
    Binds a pure tensor function (compute_func) to specific context paths, enabling:
    1. Type-safe configuration with IDE autocomplete
    2. Clean separation of pure logic from context access
    3. Simple ONNX export via path string extraction
    
    Attributes:
        compute_func: The pure tensor function that computes the observation/reward/termination.
        dynamic_vars: Maps compute_func parameter names to FieldPath objects (context paths).
                      These become ONNX inputs at export time.
        static_params: Compile-time constants (not from context, e.g., local_obs=True).
                       These are baked into the ONNX graph.
    
    Example:
        router = MdpComponent(
            compute_func=compute_mse,
            dynamic_vars={
                "a": EnvContext.current.dof_pos,      # Runtime-resolved, ONNX input
                "b": EnvContext.mimic.future_dof_pos, # Runtime-resolved, ONNX input
            },
            static_params={
                "reduction": "mean",  # Compile-time constant, baked in ONNX
            },
        )
        
        # At runtime:
        result = router.compute(ctx)  # Resolves dynamic_vars from ctx instance
        
        # For ONNX export:
        paths = router.get_bindings_dict()  # {"a": "current.dof_pos", "b": "mimic.future_dof_pos"}
    """
    
    def __init__(
        self,
        compute_func: Callable[..., Tensor],
        dynamic_vars: Dict[str, FieldPath],
        static_params: Dict[str, Any] | None = None,
    ):
        """Initialize MdpComponent.
        
        Args:
            compute_func: The pure tensor function to call.
            dynamic_vars: Maps compute_func param names to FieldPath objects (runtime-resolved).
            static_params: Compile-time constants to pass to compute_func (optional).
        """
        self.compute_func = compute_func
        self.dynamic_vars = dynamic_vars
        self.static_params = static_params or {}
        self._device_ready = False
    
    def resolve_args(self, ctx: "EnvContext") -> tuple:
        """Resolve dynamic_vars from context and prepare func params.

        Separates path resolution (pure Python, not compilable) from the
        tensor computation. ComponentManager uses this to compile only
        compute_func while keeping descriptor walks in plain Python.

        Args:
            ctx: The environment context instance to resolve paths from.

        Returns:
            Tuple of (resolved_dynamic_vars, func_params) dicts.
        """
        resolved = {}
        for param_name, field_path in self.dynamic_vars.items():
            resolved[param_name] = resolve_path(ctx, field_path.path)

        if not self._device_ready:
            self._ensure_device(resolved)

        func_params = {
            k: v for k, v in self.static_params.items()
            if k not in _METADATA_KEYS
        }
        return resolved, func_params

    def compute(self, ctx: "EnvContext") -> Tensor:
        """Resolve dynamic_vars from context and call compute_func.

        Args:
            ctx: The environment context instance to resolve paths from.

        Returns:
            The result tensor from calling compute_func(**resolved_vars, **static_params).
        """
        resolved, func_params = self.resolve_args(ctx)
        return self.compute_func(**resolved, **func_params)
    
    def get_bindings_dict(self) -> Dict[str, str]:
        """Extract path strings from FieldPath dynamic_vars for ONNX export.
        
        Returns:
            Dictionary mapping compute_func param names to dot-separated path strings.
            These become the ONNX input names.
        
        Example:
            >>> router.get_bindings_dict()
            {"body_pos": "current.rigid_body_pos", "body_rot": "current.rigid_body_rot"}
        """
        return {
            param_name: field_path.path
            for param_name, field_path in self.dynamic_vars.items()
        }
    
    def get_compute_func(self) -> Callable[..., Tensor]:
        """Get the pure tensor compute function for direct access or ONNX export.
        
        Returns:
            The compute function.
        """
        return self.compute_func
    
    def get_params(self) -> Dict[str, Any]:
        """Get static parameters (baked into ONNX graph).
        
        Returns:
            Dictionary of static parameters.
        """
        return self.static_params
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a human-readable dict for YAML/JSON config storage.
        
        Returns:
            Dictionary with compute_func name, bindings as path strings,
            and static params (tensors converted to lists).
        """
        d: Dict[str, Any] = {
            "compute_func": self.compute_func.__name__ if hasattr(self.compute_func, '__name__') else str(self.compute_func),
            "dynamic_vars": {
                param: fp.path for param, fp in self.dynamic_vars.items()
            },
        }
        # Flatten static_params into the dict (matches old config style)
        for key, val in self.static_params.items():
            if isinstance(val, Tensor):
                d[key] = val.tolist()
            else:
                d[key] = val
        return d
    
    def _ensure_device(self, resolved: Dict[str, Any]) -> None:
        """Move static_params tensors to the runtime device on first compute().
        
        Infers the target device from the first resolved Tensor binding,
        then moves any CPU tensors in static_params to that device.
        """
        # Infer device from first resolved tensor
        device = None
        for val in resolved.values():
            if isinstance(val, Tensor):
                device = val.device
                break
        
        if device is not None:
            for key, val in self.static_params.items():
                if isinstance(val, Tensor) and val.device != device:
                    self.static_params[key] = val.to(device)
        
        self._device_ready = True


def is_mdp_component(obj: Any) -> bool:
    """Check if an object is a MdpComponent instance.
    
    Args:
        obj: Object to check.
    
    Returns:
        True if obj is a MdpComponent, False otherwise.
    """
    return isinstance(obj, MdpComponent)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MdpComponent",
    "is_mdp_component",
]
