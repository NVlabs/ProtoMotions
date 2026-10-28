# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MdpComponent: binds a pure tensor kernel to context paths.

A *kernel* is a pure tensor function. Its signature IS its binding contract:
every parameter is one of

* a **context input** — its default is a ``FieldPath`` obtained from class-level
  access on ``EnvContext`` (``ref_pos: Tensor = EnvContext.mimic.ref_state.rigid_body_pos``).
  At runtime ``compute()`` resolves the path from the live context; at ONNX
  export ``get_bindings_dict()`` turns it into a named input.
* a **constant** — its default is a plain value (a scalar, a fixed tensor).
* **required** — no default; a recipe must supply it through an override.

A recipe binds a kernel with one line::

    MdpComponent.from_kernel(my_kernel, meta=RewardMeta(weight=0.5), coefficient=-2.0)

``from_kernel`` walks the signature: a ``FieldPath`` default becomes a dynamic
binding, any other default a constant, a missing default must be overridden. An
override that is a ``FieldPath`` rebinds that input; any other value (``None``
included) sets a constant and so unbinds a ``FieldPath`` default. A misspelled
override, a parameter with neither default nor override, a ``functools.partial``
and a ``*args``/``**kwargs`` kernel each fail at construction naming the kernel
and the parameter.

A directly constructed ``MdpComponent(kernel, dynamic_vars, static_params)``
reads the signature the same way for what it leaves unbound: a ``FieldPath``
default that is in neither dict is bound to its path, so a kernel that declares
``anchor_body_index: int = EnvContext.current.anchor_idx`` reads the published
anchor however the recipe built the component, never the path object.

Reward metadata is a typed ``RewardMeta`` (``protomotions.envs.rewards.meta``)
passed as ``meta=``; it is never a ``static_params`` key, so a kernel parameter
may be named ``weight`` or ``threshold`` freely. ``static_params`` reach the
kernel verbatim.

Eagerness is a marker on the kernel: a kernel that must not be ``torch.compile``d
is decorated ``@eager_kernel``. The component carries no compile flag.
Exportability is a marker too: a kernel the deployable graph cannot carry (it
needs the whole terrain height field) is decorated ``@non_exportable``, and the
exporter refuses an actor observation bound to it.

A kernel that reads a window (``[envs, steps, ...]``) says which slots with a
parameter annotated ``StepSelection`` (``protomotions.envs.steps``). The
component normalises the bound value to a tuple of 1-based steps at
construction (an int ``N`` is ``1..N``) and refuses a context path there, so a
selection is always a constant; ``step_selections()`` reports them for the
window check and the exporter. The kernel's annotations are read at runtime
(``typing.get_type_hints``), so they must resolve.

Example
-------
::

    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.rewards.meta import RewardMeta

    def compute_soft_pos_limit_rew(
        dof_pos: Tensor = EnvContext.current.dof_pos,
        dof_limits_lower: Tensor = ...,
        dof_limits_upper: Tensor = ...,
    ) -> Tensor:
        ...

    reward_components = {
        "limits_dof_pos": MdpComponent.from_kernel(
            compute_soft_pos_limit_rew,
            meta=RewardMeta(weight=-100.0),
            dof_limits_lower=robot_cfg.kinematic_info.dof_limits_lower,
            dof_limits_upper=robot_cfg.kinematic_info.dof_limits_upper,
        ),
    }
"""

from __future__ import annotations

import functools
import inspect
import typing
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, TYPE_CHECKING

import torch
from torch import Tensor

from protomotions.envs.context_paths import FieldPath, resolve_path
from protomotions.envs.steps import StepSelection, normalize_steps

if TYPE_CHECKING:
    from protomotions.envs.context import FrozenContext
    from protomotions.envs.rewards.meta import RewardMeta


_OPAQUE_KERNEL_MESSAGE = (
    "MdpComponent refuses functools.partial and uninspectable callables: write a "
    "kernel and pass its constants as parameters ({!r})"
)


def _refuse_opaque_kernel(func: Callable) -> None:
    """Refuse a ``functools.partial`` or a callable ``inspect`` cannot read.

    A kernel's signature is its binding contract, so it must be inspectable and
    name every input. A partial hides bound arguments; a TorchScript function or
    a C callable has no readable signature.
    """
    if isinstance(func, functools.partial):
        raise TypeError(_OPAQUE_KERNEL_MESSAGE.format(func))
    try:
        inspect.signature(func)
    except (TypeError, ValueError) as error:
        raise TypeError(_OPAQUE_KERNEL_MESSAGE.format(func)) from error


def _stable_callable_name(func: Callable) -> str:
    """Name a compute function the same way in every process.

    ``to_dict`` output is saved with the run's configs and hashed to compare
    configs across ranks and restarts, so the name must not depend on the
    process (no memory addresses): a function by its ``__name__``, any other
    callable by its class.
    """
    return getattr(func, "__name__", None) or type(func).__qualname__


def _kernel_type_hints(kernel: Callable) -> Dict[str, Any]:
    """The resolved annotations of a kernel.

    A function's own; a callable object's are those of its ``__call__``. A
    hint that does not resolve (a name imported only under ``TYPE_CHECKING``,
    a forward reference to nothing) is refused naming the kernel: the
    signature is the binding contract and is read at runtime.
    """
    target = kernel if inspect.isroutine(kernel) else type(kernel).__call__
    try:
        return typing.get_type_hints(target)
    except (NameError, TypeError, AttributeError, SyntaxError) as error:
        raise TypeError(
            f"kernel {_stable_callable_name(kernel)}: an annotation does not "
            f"resolve ({error}); a kernel's hints are read at runtime, so the "
            "names they use must be importable"
        ) from error


def _step_parameters(kernel: Callable) -> Dict[str, bool]:
    """The kernel parameters annotated ``StepSelection`` -> whether None is allowed.

    ``history_steps: StepSelection`` maps to False, ``future_steps:
    Optional[StepSelection]`` to True.
    """
    out: Dict[str, bool] = {}
    for name, hint in _kernel_type_hints(kernel).items():
        if name == "return":
            continue
        if hint is StepSelection:
            out[name] = False
        elif typing.get_origin(
            hint
        ) is typing.Union and StepSelection in typing.get_args(hint):
            out[name] = type(None) in typing.get_args(hint)
    return out


def eager_kernel(fn: Callable) -> Callable:
    """Mark a kernel to run eagerly: ``ComponentManager`` never ``torch.compile``s it.

    Use on a kernel whose shared, rank-polymorphic casts would exhaust Dynamo's
    specialization cache, or that cannot be traced. ONNX export can still trace
    the underlying tensor ops.
    """
    fn.run_eagerly = True
    return fn


def is_eager_kernel(kernel: Callable) -> bool:
    """Whether ``kernel`` is marked ``@eager_kernel`` (so it is never compiled)."""
    return getattr(kernel, "run_eagerly", False)


def non_exportable(fn: Callable) -> Callable:
    """Mark a kernel the deployable graph cannot carry: the exporter refuses it.

    Use on a kernel that needs the whole of something the deployed policy never
    holds (the terrain height field); its observation reaches the deployed
    policy as a passthrough input instead. Training and inference in simulation
    run it like any other kernel.
    """
    fn.exportable = False
    return fn


def is_exportable(kernel: Callable) -> bool:
    """Whether ``kernel`` may be traced into the deployable graph (the default)."""
    return getattr(kernel, "exportable", True)


class MdpComponent:
    """A pure tensor kernel with explicit context bindings.

    Attributes:
        compute_func: The pure tensor kernel.
        dynamic_vars: Maps kernel parameter names to ``FieldPath`` objects
            (context paths); these become ONNX inputs at export time.
        static_params: Constants passed to the kernel verbatim (baked into the
            ONNX graph). Fixed tensors are moved to the runtime device on the
            first ``compute()`` call.
        meta: A ``RewardMeta`` for a reward component, else ``None``.
        required_keys: Required keys in mapping-valued dynamic arguments,
            checked by ``ComponentManager`` when the environment is built.
    """

    def __init__(
        self,
        compute_func: Callable[..., Tensor],
        dynamic_vars: Dict[str, FieldPath],
        static_params: Dict[str, Any] | None = None,
        *,
        meta: Optional["RewardMeta"] = None,
        required_keys: Dict[str, list[str]] | None = None,
    ):
        """Initialize MdpComponent.

        Args:
            compute_func: The pure tensor kernel to call.
            dynamic_vars: Maps kernel param names to ``FieldPath`` objects.
            static_params: Constants passed to the kernel (optional).
            meta: Reward metadata for a reward component (optional).
            required_keys: Mapping argument name -> required names or glob
                patterns. Each pattern must match at least one registered key.
        """
        _refuse_opaque_kernel(compute_func)
        self.compute_func = compute_func
        # Both dicts are the component's own: the recipe's are never mutated by
        # the default binding or the step normalisation below.
        self.dynamic_vars = dict(dynamic_vars)
        self.static_params = dict(static_params or {})
        self._bind_path_defaults()
        self.meta = meta
        # Required entries in mapping-valued dynamic arguments, checked at setup.
        self.required_keys = required_keys or {}
        unbound = self.required_keys.keys() - self.dynamic_vars.keys()
        if unbound:
            raise ValueError(
                f"Required mapping keys refer to unbound arguments: {unbound}"
            )
        for argument, keys in self.required_keys.items():
            # An empty list requires nothing (a fall check with no forbidden
            # body); a bare string or a repeated name is a recipe mistake.
            if isinstance(keys, str) or len(set(keys)) != len(keys):
                raise ValueError(
                    f"required keys for '{argument}' must be a list of distinct "
                    f"names or patterns, got {keys!r}"
                )
        self._normalize_step_selections()
        self._device_ready = False

    def _bind_path_defaults(self) -> None:
        """Bind every ``FieldPath`` default the constructor left unbound.

        What ``from_kernel`` does for every parameter, done here for the ones a
        direct construction named in neither dict: the signature is the
        binding contract whichever way the component was built, so a kernel
        never receives the path object in place of the published value.
        """
        for name, parameter in inspect.signature(self.compute_func).parameters.items():
            if (
                isinstance(parameter.default, FieldPath)
                and name not in self.dynamic_vars
                and name not in self.static_params
            ):
                self.dynamic_vars[name] = parameter.default

    def _normalize_step_selections(self) -> None:
        """Turn every bound ``StepSelection`` parameter into its tuple of steps.

        A context path bound to one is refused: a selection is a constant the
        kernel is compiled and exported with, never a value read at runtime. A
        selection with no default that the recipe left unbound is refused here
        too, so a forgotten ``history_steps`` fails when the component is built
        (and when a saved configuration is loaded), not at the first call.
        """
        name = _stable_callable_name(self.compute_func)
        parameters = inspect.signature(self.compute_func).parameters
        for parameter, optional in _step_parameters(self.compute_func).items():
            if parameter in self.dynamic_vars:
                raise TypeError(
                    f"kernel {name}: '{parameter}' is a StepSelection and must be a "
                    f"constant, not the context path "
                    f"'{self.dynamic_vars[parameter].path}'"
                )
            if parameter not in self.static_params:
                if parameters[parameter].default is inspect.Parameter.empty:
                    raise TypeError(
                        f"kernel {name}: '{parameter}' is a StepSelection with no "
                        "default and must be bound as a static parameter"
                    )
                continue
            value = self.static_params[parameter]
            if optional and value is None:
                continue
            try:
                self.static_params[parameter] = normalize_steps(value)
            except (TypeError, ValueError) as error:
                raise type(error)(
                    f"kernel {name}, parameter '{parameter}': {error}"
                ) from error

    def step_selections(self) -> Dict[str, Optional[Tuple[int, ...]]]:
        """The kernel's bound step selections by parameter name.

        A normalised tuple of 1-based steps, or ``None`` for an
        ``Optional[StepSelection]`` left at ``None`` (the whole window): the
        bound constant, else the kernel's own default. What the window check
        compares against the published slots and what the exporter sizes its
        history inputs from.
        """
        parameters = inspect.signature(self.compute_func).parameters
        selections: Dict[str, Optional[Tuple[int, ...]]] = {}
        for parameter in _step_parameters(self.compute_func):
            if parameter in self.static_params:
                selections[parameter] = self.static_params[parameter]
            elif parameters[parameter].default is not inspect.Parameter.empty:
                default = parameters[parameter].default
                selections[parameter] = (
                    None if default is None else normalize_steps(default)
                )
        return selections

    @classmethod
    def from_kernel(
        cls,
        kernel: Callable[..., Tensor],
        *,
        meta: Optional["RewardMeta"] = None,
        required_keys: Dict[str, list[str]] | None = None,
        **overrides: Any,
    ) -> "MdpComponent":
        """Bind ``kernel`` by reading its signature.

        A ``FieldPath`` default is a context input, a plain default a constant,
        a missing default must be supplied in ``overrides``. An override that is
        a ``FieldPath`` rebinds; any other value (``None`` included) is a
        constant and unbinds a ``FieldPath`` default.
        """
        _refuse_opaque_kernel(kernel)
        name = _stable_callable_name(kernel)
        parameters = inspect.signature(kernel).parameters

        dynamic_vars: Dict[str, FieldPath] = {}
        static_params: Dict[str, Any] = {}
        for param_name, param in parameters.items():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                raise TypeError(
                    f"kernel {name} declares *args/**kwargs; a kernel names every input"
                )
            if param_name in overrides:
                continue
            default = param.default
            if default is inspect.Parameter.empty:
                raise TypeError(
                    f"kernel {name}: parameter {param_name} has no default and "
                    "no override"
                )
            if isinstance(default, FieldPath):
                dynamic_vars[param_name] = default
            else:
                static_params[param_name] = default

        for key, value in overrides.items():
            if key not in parameters:
                raise TypeError(
                    f"kernel {name} has no parameter {key} (parameters: "
                    f"{', '.join(parameters)})"
                )
            if isinstance(value, FieldPath):
                dynamic_vars[key] = value
            else:
                static_params[key] = value

        return cls(
            kernel,
            dynamic_vars,
            static_params,
            meta=meta,
            required_keys=required_keys,
        )

    def rebind(self, **paths: FieldPath) -> "MdpComponent":
        """Copy this component with the named dynamic bindings replaced.

        Each name must already be a dynamic binding; this swaps the path it
        reads (``sensed`` uses it to point an actor's observations at a sensor
        source). The kernel, metadata and required keys are shared; the
        ``static_params`` dict is copied so the copy's first-compute device move
        does not mutate the original's.
        """
        unknown = set(paths) - set(self.dynamic_vars)
        if unknown:
            raise ValueError(
                f"rebind: {sorted(unknown)} are not dynamic bindings of "
                f"{_stable_callable_name(self.compute_func)}"
            )
        new_dynamic = dict(self.dynamic_vars)
        new_dynamic.update(paths)
        return MdpComponent(
            self.compute_func,
            new_dynamic,
            dict(self.static_params),
            meta=self.meta,
            required_keys=self.required_keys,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_device_ready", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._device_ready = False

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MdpComponent):
            return NotImplemented
        return (
            self.compute_func == other.compute_func
            and self.dynamic_vars == other.dynamic_vars
            and self._static_params_equal(other.static_params)
            and self.meta == other.meta
            and self.required_keys == other.required_keys
        )

    __hash__ = None

    def _static_params_equal(self, other: Dict[str, Any]) -> bool:
        if self.static_params.keys() != other.keys():
            return False
        for key, value in self.static_params.items():
            other_value = other[key]
            if isinstance(value, Tensor) or isinstance(other_value, Tensor):
                if not (
                    isinstance(value, Tensor)
                    and isinstance(other_value, Tensor)
                    and value.shape == other_value.shape
                    and torch.equal(value, other_value)
                ):
                    return False
            elif value != other_value:
                return False
        return True

    def resolve_args(self, ctx: "FrozenContext") -> tuple:
        """Resolve dynamic_vars from context; return ``(resolved, static_params)``.

        Separates path resolution (pure Python, not compilable) from the tensor
        computation. ``static_params`` reach the kernel verbatim.
        """
        resolved = {
            param_name: resolve_path(ctx, field_path.path)
            for param_name, field_path in self.dynamic_vars.items()
        }
        if not self._device_ready:
            self._ensure_device(resolved)
        return resolved, self.static_params

    def compute(self, ctx: "FrozenContext") -> Tensor:
        """Resolve dynamic_vars from context and call compute_func."""
        resolved, func_params = self.resolve_args(ctx)
        return self.compute_func(**resolved, **func_params)

    def get_bindings_dict(self) -> Dict[str, str]:
        """Path strings of the dynamic bindings, for ONNX export.

        Example:
            >>> router.get_bindings_dict()
            {"body_pos": "current.rigid_body_pos", "body_rot": "current.rigid_body_rot"}
        """
        return {
            param_name: field_path.path
            for param_name, field_path in self.dynamic_vars.items()
        }

    def get_compute_func(self) -> Callable[..., Tensor]:
        """The pure tensor kernel, for direct access or ONNX export."""
        return self.compute_func

    def get_params(self) -> Dict[str, Any]:
        """Static parameters (baked into the ONNX graph)."""
        return self.static_params

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a human-readable dict for YAML/JSON config storage."""
        d: Dict[str, Any] = {
            "compute_func": _stable_callable_name(self.compute_func),
            "dynamic_vars": {param: fp.path for param, fp in self.dynamic_vars.items()},
        }
        if self.meta is not None:
            d["meta"] = asdict(self.meta)
        if self.required_keys:
            d["required_keys"] = self.required_keys
        # Flatten static_params into the dict (matches old config style).
        for key, val in self.static_params.items():
            if isinstance(val, Tensor):
                d[key] = val.tolist()
            elif is_dataclass(val) and not isinstance(val, type):
                # A declarative static (a BodySelection): its fields, the names
                # the recipe wrote, not what the build resolves them to.
                d[key] = asdict(val)
            else:
                d[key] = val
        return d

    def _ensure_device(self, resolved: Dict[str, Any]) -> None:
        """Move static_params tensors to the runtime device on first compute().

        Infers the target device from the first resolved Tensor binding, then
        moves any CPU tensors in static_params to that device.
        """
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


# Names that are reward metadata (now on RewardMeta) or evaluation/termination
# rule settings, never kernel parameters: a stale one in static_params is
# flagged with a pointer to its real home.
_REWARD_META_NAMES = frozenset(
    {"weight", "multiplicative", "min_value", "max_value", "zero_during_grace_period"}
)
_RULE_SETTING_NAMES = frozenset({"threshold", "fail_above", "use_region_weights"})


def _stale_key_hint(key: str) -> str:
    if key in _REWARD_META_NAMES:
        return (
            f"; reward metadata like '{key}' goes on meta=RewardMeta(...), not "
            "static_params"
        )
    if key in _RULE_SETTING_NAMES:
        return (
            f"; '{key}' is an evaluation or termination rule setting, not a kernel "
            "parameter"
        )
    return ""


def _signature_errors(
    name: str, component: "MdpComponent", group: str, *, runtime_inputs: bool = False
) -> List[str]:
    """Check a component's bindings against its kernel's signature.

    Every static_param and dynamic binding must be a kernel parameter, and every
    parameter with no default must be bound. ``from_kernel`` builds this way by
    construction; this is the same exact check for a directly constructed
    ``MdpComponent``, so a stale key (an old ``weight``/``threshold``, a typo)
    is refused at build with the kernel named, instead of failing as a
    ``TypeError`` on the first ``compute()``. With ``runtime_inputs`` the
    table's owner supplies the parameters the bindings leave out when it calls
    the kernel (the AMP reference table hands its kernels the motion library's
    state), so only a bound name the kernel does not take is an error.
    """
    kernel = component.compute_func
    try:
        parameters = inspect.signature(kernel).parameters
    except (TypeError, ValueError):
        return []  # uninspectable: the constructor already refuses these
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return []  # a **kwargs kernel accepts any name; nothing to check
    kernel_name = _stable_callable_name(kernel)
    bound = set(component.static_params) | set(component.dynamic_vars)
    errors = []
    for key in sorted(bound - set(parameters)):
        errors.append(
            f"{group} component '{name}': '{key}' is not a parameter of kernel "
            f"'{kernel_name}'" + _stale_key_hint(key)
        )
    if runtime_inputs:
        return errors
    for param_name, parameter in parameters.items():
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        if parameter.default is inspect.Parameter.empty and param_name not in bound:
            errors.append(
                f"{group} component '{name}': kernel '{kernel_name}' parameter "
                f"'{param_name}' is required but not bound"
            )
    return errors


def static_component_errors(
    components: Mapping[str, "MdpComponent"],
    *,
    group: str,
    runtime_inputs: bool = False,
) -> List[str]:
    """Metadata-placement and binding-against-signature errors for one group.

    A reward component must carry a ``RewardMeta`` and a component of any other
    group must not; and every component's static_params and dynamic bindings
    must match its kernel's signature (:func:`_signature_errors`, whose
    ``runtime_inputs`` this is: the table's owner supplies the unbound
    parameters at the call). Called by ``ComponentManager.validate_bindings``
    at build and by ``context_check.check_bindings_against_config`` from the
    configs alone, so both word the refusal identically.
    """
    errors: List[str] = []
    for name, component in components.items():
        has_meta = getattr(component, "meta", None) is not None
        if group == "reward":
            if not has_meta:
                errors.append(
                    f"reward component '{name}' has no RewardMeta: pass "
                    "meta=RewardMeta(weight=...) to from_kernel"
                )
        elif has_meta:
            errors.append(
                f"{group} component '{name}' carries a RewardMeta; only reward "
                "components take meta"
            )
        errors.extend(
            _signature_errors(name, component, group, runtime_inputs=runtime_inputs)
        )
    return errors


def is_mdp_component(obj: Any) -> bool:
    """Check if an object is a MdpComponent instance."""
    return isinstance(obj, MdpComponent)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MdpComponent",
    "eager_kernel",
    "is_eager_kernel",
    "is_mdp_component",
    "static_component_errors",
]
