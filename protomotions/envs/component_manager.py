# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Component execution infrastructure.

Pure infra: iterates MdpComponent configs, compiles compute methods, executes kernels.
Users debugging reward or termination behavior should look at base_env/mdp_managers.py instead.

This module handles:
- torch.compile caching for performance
- MdpComponent iteration and execution
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from fnmatch import fnmatchcase
from typing import Any, Callable, Dict, TYPE_CHECKING

import torch
from torch import Tensor

from protomotions.envs.body_selection import unresolved_selections
from protomotions.envs.context_paths import resolve_path
from protomotions.envs.mdp_component import (
    _stable_callable_name,
    is_eager_kernel,
    static_component_errors,
)
from protomotions.envs.steps import is_window_path

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from protomotions.envs.context import FrozenContext
    from protomotions.envs.mdp_component import MdpComponent

# torch.compile unavailable on Python 3.8 (IsaacGym)
TORCH_COMPILE_AVAILABLE = hasattr(torch, "compile") and sys.version_info >= (3, 9)


class ComponentManager:
    """Executes MdpComponent components with torch.compile caching.

    Pure infrastructure class - handles iteration, compilation, and calling.
    Does not contain any RL-specific logic (see base_env/mdp_managers.py for that).

    Usage:
        from protomotions.envs.mdp_component import MdpComponent
        from protomotions.envs.context_views import EnvContext

        manager = ComponentManager(device)
        raw_results = manager.execute_all(
            components={
                "obs1": MdpComponent(
                    compute_func=compute_kernel,
                    dynamic_vars={"param": EnvContext.current.dof_pos},
                )
            },
            ctx=env_context,
        )
    """

    def __init__(self, device: torch.device):
        """Initialize component manager.

        Args:
            device: Device for tensor operations.
        """
        self.device = device
        self._compiled: Dict[tuple[Callable, bool], Callable] = {}

    def validate_bindings(
        self,
        components: Dict[str, "MdpComponent"],
        ctx: "FrozenContext",
        *,
        group: str,
    ) -> None:
        """Resolve bindings at setup, before compiling or executing any kernel.

        Availability is determined by the actual context, not a second list of
        sensor capabilities. Explicit None values remain valid optional inputs.
        Reward metadata placement (``static_component_errors``) is checked first,
        in the same wording the config-only check uses. A ``BodySelection``
        still in a component's static parameters is refused (``build_env``
        resolves selections before the managers are built), and a step
        selection that reaches past the slots a window path publishes is
        refused with the path and its slot count.
        """
        meta_errors = static_component_errors(components, group=group)
        if meta_errors:
            raise ValueError("\n".join(meta_errors))
        for name, component in components.items():
            for parameter in unresolved_selections(component):
                raise ValueError(
                    f"{group} component '{name}', parameter '{parameter}': "
                    "BodySelection was not resolved; build_env resolves "
                    "selections before the managers are built"
                )

        # path -> required patterns demanded across components, and the mapping's
        # measured keys, for the reverse coverage warning below.
        demanded: Dict[str, set] = {}
        measured: Dict[str, list] = {}
        for name, component in components.items():
            selections = component.step_selections()
            for argument, path in component.dynamic_vars.items():
                try:
                    value = resolve_path(ctx, path.path)
                except AttributeError as error:
                    raise ValueError(
                        f"{group} component '{name}', argument '{argument}': "
                        f"context path '{path.path}' is unavailable"
                    ) from error
                if (
                    is_window_path(path.path)
                    and isinstance(value, Tensor)
                    and value.ndim >= 2
                ):
                    for steps in selections.values():
                        if steps is not None and max(steps) > value.shape[1]:
                            raise ValueError(
                                f"{group} component '{name}', argument '{argument}': "
                                f"step selection {steps} reaches slot {max(steps)} "
                                f"but '{path.path}' publishes {value.shape[1]} slots"
                            )
                required_keys = component.required_keys.get(argument, [])
                if required_keys and not isinstance(value, Mapping):
                    raise ValueError(
                        f"{group} component '{name}', argument '{argument}': "
                        f"required keys need a mapping at '{path.path}'"
                    )
                for key in required_keys:
                    if not any(fnmatchcase(name, key) for name in value):
                        raise ValueError(
                            f"{group} component '{name}', argument '{argument}': "
                            f"required key '{key}' is not registered in '{path.path}'"
                        )
                if required_keys:
                    demanded.setdefault(path.path, set()).update(required_keys)
                    measured.setdefault(path.path, list(value))

        # Reverse coverage: a measured contact row or partner column that no
        # bound component names is likely a wasted measurement. Warn, do not
        # fail: a configuration may measure more than any one recipe reads.
        for path, keys in measured.items():
            patterns = demanded[path]
            unnamed = [
                key
                for key in keys
                if not any(fnmatchcase(key, pattern) for pattern in patterns)
            ]
            if unnamed:
                log.warning(
                    "%s components: '%s' measures %s, which no bound component names",
                    group,
                    path,
                    unnamed,
                )

    def execute_all(
        self,
        components: Dict[str, "MdpComponent"],
        ctx: "FrozenContext",
    ) -> Dict[str, Tensor]:
        """Execute all MdpComponent components, return raw results dict.

        Path resolution (descriptor walks) happens in plain Python.
        Only the pure tensor compute_func is torch.compiled.

        Args:
            components: Dict of {name: MdpComponent}.
            ctx: Environment context passed to all routers.

        Returns:
            Dict of {name: result_tensor} for each component.
        """
        # Reset subsets have variable batch sizes; keep them out of the stable
        # all-environment compile cache. Single and bulk calls share execution.
        compile_components = getattr(ctx, "env_ids", None) is None
        return {
            name: self.execute_single(name, router, ctx, compile=compile_components)
            for name, router in components.items()
        }

    def execute_single(
        self,
        name: str,
        router: "MdpComponent",
        ctx: "FrozenContext",
        compile: bool = False,
    ) -> Any:
        """Execute a single MdpComponent component.

        Args:
            name: Component name (for diagnostics).
            router: MdpComponent instance.
            ctx: Environment context.
            compile: Whether to use torch.compile (default False for single calls).

        Returns:
            Result tensor from the kernel.
        """
        if not compile:
            return router.compute(ctx)
        resolved, func_params = router.resolve_args(ctx)
        compiled_fn = self._get_compiled_func(router)
        if compiled_fn is router.compute_func:
            # torch.compile unavailable, or an @eager_kernel: the kernel's own
            # exception propagates as it is.
            return compiled_fn(**resolved, **func_params)
        try:
            return compiled_fn(**resolved, **func_params)
        except Exception as exc:
            # Surfaces on this call because torch.compile compiles lazily: the
            # backend failing (triton or a C++ compiler missing, an inductor
            # bug), dynamo unable to trace this kernel, or the kernel's own
            # error. Named once, chained to the original so its traceback is
            # kept; nothing is demoted to eager and nothing re-runs. The import
            # sits here because this path only runs with torch.compile in play.
            from torch._dynamo.exc import BackendCompilerFailed

            kernel = _stable_callable_name(router.compute_func)
            if isinstance(exc, BackendCompilerFailed):
                hint = (
                    "torch.compile's backend failed, so every compiled kernel "
                    "fails the same way: the environment, not the kernel (triton "
                    "or a C++ compiler missing, an inductor bug); see the chained "
                    "error."
                )
            else:
                hint = (
                    "A kernel error: fix the kernel; if only dynamo's tracing of "
                    "this kernel fails, mark it @eager_kernel."
                )
            raise RuntimeError(
                f"component '{name}': kernel '{kernel}' raised under torch.compile "
                f"({type(exc).__name__}: {exc}). {hint}"
            ) from exc

    @staticmethod
    def _cache_key(router: "MdpComponent") -> tuple:
        """Key the compiled-function cache by kernel and its eager marker.

        Names are local to observation/reward/termination groups; keying by the
        kernel (and whether it is ``@eager_kernel``) so equal names cannot select
        another function or make an explicitly eager kernel use a compiled one.
        """
        return (router.compute_func, is_eager_kernel(router.compute_func))

    def _get_compiled_func(self, router: "MdpComponent") -> Callable:
        """The callable for an all-env context: the kernel torch.compiled once
        and cached, or the kernel itself when torch.compile is unavailable
        (Python 3.8) or the kernel is ``@eager_kernel``.

        Only the pure tensor kernel (compute_func) is compiled, not the path
        resolution, so torch.dynamo never traces through FieldPath descriptors.
        torch.compile compiles lazily, on the first call: an error there (the
        kernel's own, a tracing failure, or the backend failing) propagates
        from ``execute_single`` with the component named. The manager never
        falls back to eager behind the caller's back (dynamo's own
        recompile-limit fallback is torch's default and unchanged); a kernel
        torch.compile cannot trace is marked ``@eager_kernel`` instead (open
        item 9's decision).

        Args:
            router: MdpComponent instance.

        Returns:
            The compiled kernel, or the kernel.
        """
        cache_key = self._cache_key(router)
        if cache_key not in self._compiled:
            fn = router.compute_func
            if not TORCH_COMPILE_AVAILABLE or is_eager_kernel(fn):
                self._compiled[cache_key] = fn
            else:
                self._compiled[cache_key] = torch.compile(fn, mode="default")
        return self._compiled[cache_key]

    def clear_cache(self):
        """Clear compiled function cache."""
        self._compiled.clear()
