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
"""Component execution infrastructure.

Pure infra: iterates MdpComponent configs, compiles compute methods, executes kernels.
Users debugging RL behavior should look at base_env/utils.py instead.

This module handles:
- torch.compile caching for performance
- MdpComponent iteration and execution
"""

import logging
import sys
from typing import Any, Callable, Dict, TYPE_CHECKING

import torch
from torch import Tensor

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent

# torch.compile unavailable on Python 3.8 (IsaacGym)
TORCH_COMPILE_AVAILABLE = hasattr(torch, "compile") and sys.version_info >= (3, 9)


class ComponentManager:
    """Executes MdpComponent components with torch.compile caching.
    
    Pure infrastructure class - handles iteration, compilation, and calling.
    Does not contain any RL-specific logic (see base_env/utils.py for that).
    
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
        self._compiled: Dict[str, Callable] = {}
    
    def execute_all(
        self,
        components: Dict[str, "MdpComponent"],
        ctx: "EnvContext",
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
        results = {}
        for name, router in components.items():
            resolved, func_params = router.resolve_args(ctx)
            compiled_fn = self._get_compiled_func(name, router)
            try:
                results[name] = compiled_fn(**resolved, **func_params)
            except Exception:
                # torch.compile is lazy — Inductor errors surface at call time.
                # Fall back to the uncompiled function and cache it.
                log.warning("torch.compile failed for '%s', falling back to eager mode", name)
                fn = router.compute_func
                self._compiled[f"{name}_func"] = fn
                results[name] = fn(**resolved, **func_params)
        return results

    def execute_single(
        self,
        name: str,
        router: "MdpComponent",
        ctx: "EnvContext",
        compile: bool = False,
    ) -> Any:
        """Execute a single MdpComponent component.

        Args:
            name: Component name (for caching).
            router: MdpComponent instance.
            ctx: Environment context.
            compile: Whether to use torch.compile (default False for single calls).

        Returns:
            Result tensor from the kernel.
        """
        if compile:
            resolved, func_params = router.resolve_args(ctx)
            compiled_fn = self._get_compiled_func(name, router)
            try:
                return compiled_fn(**resolved, **func_params)
            except Exception:
                log.warning("torch.compile failed for '%s', falling back to eager mode", name)
                fn = router.compute_func
                self._compiled[f"{name}_func"] = fn
                return fn(**resolved, **func_params)
        else:
            return router.compute(ctx)

    def _get_compiled_func(self, name: str, router: "MdpComponent") -> Callable:
        """Get torch.compiled version of compute_func, caching result.

        Compiles only the pure tensor kernel (compute_func), not the path
        resolution logic. This avoids torch.dynamo tracing through FieldPath
        descriptors while still compiling all GPU work.

        torch.compile uses lazy compilation — the actual Triton/Inductor
        compilation happens on the first *call*, not at torch.compile() time.
        To handle this, we wrap compiled functions so that if the first call
        triggers a compilation error, we transparently fall back to eager mode
        and cache the uncompiled version for all future calls.

        Args:
            name: Component name for caching.
            router: MdpComponent instance.

        Returns:
            Compiled compute_func callable.
        """
        cache_key = f"{name}_func"
        if cache_key not in self._compiled:
            fn = router.compute_func
            if not TORCH_COMPILE_AVAILABLE:
                self._compiled[cache_key] = fn
            else:
                try:
                    compiled_fn = torch.compile(fn, mode="default")
                except Exception:
                    compiled_fn = None

                if compiled_fn is not None:
                    # Wrap to catch lazy compilation failures (Triton/Inductor
                    # errors that surface on first call). After first successful
                    # call, the wrapper replaces itself with the compiled version
                    # so there is no overhead on subsequent calls.
                    def _fallback_wrapper(
                        *args,
                        _key=cache_key,
                        _compiled=compiled_fn,
                        _eager=fn,
                        **kwargs,
                    ):
                        try:
                            result = _compiled(*args, **kwargs)
                            # Compilation succeeded — promote to compiled version
                            self._compiled[_key] = _compiled
                            return result
                        except Exception:
                            # Compilation failed — fall back to eager permanently
                            self._compiled[_key] = _eager
                            return _eager(*args, **kwargs)

                    self._compiled[cache_key] = _fallback_wrapper
                else:
                    self._compiled[cache_key] = fn
        return self._compiled[cache_key]
    
    def clear_cache(self):
        """Clear compiled function cache."""
        self._compiled.clear()
