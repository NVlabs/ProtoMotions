# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step selections: which slots of a published window a kernel reads.

A *window* is a context tensor published as ``[envs, steps, ...]``: the state
history (``historical.*`` and its sensor twins), the two action rings
(``raw_action_history``, ``processed_action_history``), the mimic future
targets (``mimic.future_*``) and the masked-mimic target streams
(``masked_mimic.ref_*``, ``masked_mimic.local_ref_*``, their per-step masks
and times). A kernel that reads a window declares which 1-based slots it takes
with a parameter annotated :data:`StepSelection`:

* ``history_steps: StepSelection`` with no default: the recipe writes the
  steps (``history_steps=[1, 2, 4]`` or ``history_steps=8`` for ``1..8``);
* ``future_steps: Optional[StepSelection] = None``: ``None`` reads the whole
  published horizon.

``MdpComponent`` normalises the recipe's value to a tuple at construction with
:func:`normalize_steps` and refuses a context path there, so a selection is
always a compile-time constant; ``ComponentManager`` checks at build that no
selection reaches past the window it is bound to (:data:`WINDOW_PREFIXES`).
The kernel selects with :func:`select_steps` and ages the selected slots with
:func:`step_ages`, naming the time origin instead of inlining one of the two
formulas the old kernels carried. :func:`required_history_depth` is the one
computation of how many past slots a table of components needs from the
history ring; the ring and the config-only check both call it.

This module imports nothing from the kernel packages, so ``mdp_component`` can
import it.
"""

from __future__ import annotations

import numbers
from typing import (
    Callable,
    Collection,
    Dict,
    Iterable,
    Mapping,
    NewType,
    Optional,
    Protocol,
    Tuple,
)

import torch
from torch import Tensor

from protomotions.envs.context_sources import HISTORICAL_NAMESPACES

# A kernel parameter annotation: the 1-based window slots the kernel reads. Used
# only as an annotation; the bound value is a tuple once MdpComponent has
# normalised it.
StepSelection = NewType("StepSelection", Tuple[int, ...])

_NOT_A_SELECTION = "a step selection is an int N or a sequence of ints, not {!r}"

# Context paths published as [envs, steps, ...]; a step selection past ``steps``
# on a path under one of these prefixes is refused at build.
WINDOW_PREFIXES: Tuple[str, ...] = (
    *(f"{namespace}." for namespace in HISTORICAL_NAMESPACES),
    "raw_action_history",
    "processed_action_history",
    "mimic.future_",
    "masked_mimic.ref_",
    "masked_mimic.local_ref_",
    "masked_mimic.time_offsets",
    "masked_mimic.local_time_offsets",
    "masked_mimic.target_poses_masks",
    "masked_mimic.local_target_poses_masks",
    "masked_mimic.anchor_translation_mask",
    "masked_mimic.anchor_rotation_mask",
)

# The two ways a kernel ages a selected slot k: ``current`` is k * dt (the
# absolute age the proprioception tokens carry), ``first_slot`` is (k - 1) * dt
# (the pose-token convention: the first stored slot has age zero).
_TIME_ORIGINS: Dict[str, float] = {"current": 0.0, "first_slot": 1.0}


def is_window_path(path: str) -> bool:
    """Whether ``path`` is published as ``[envs, steps, ...]``."""
    return path.startswith(WINDOW_PREFIXES)


def normalize_steps(value) -> Tuple[int, ...]:
    """The tuple of 1-based steps a recipe's value means.

    An int ``N`` is ``(1, ..., N)``; a sequence is checked (non-empty, positive,
    unique ints) and kept in the order written. ``None`` is refused: a kernel
    that reads a window says which slots.
    """
    if value is None:
        raise TypeError(
            "a step selection cannot be None: write the steps this kernel reads"
        )
    if isinstance(value, bool):
        raise TypeError(_NOT_A_SELECTION.format(value))
    if isinstance(value, numbers.Integral):
        if value < 1:
            raise ValueError(
                f"a step selection N means the steps 1..N, so N must be positive, "
                f"not {int(value)}"
            )
        return tuple(range(1, int(value) + 1))
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(_NOT_A_SELECTION.format(value))
    steps = tuple(value)
    if not steps:
        raise ValueError("a step selection cannot be empty")
    for step in steps:
        if isinstance(step, bool) or not isinstance(step, numbers.Integral):
            raise TypeError(
                f"step selection {steps!r}: every step is an int, not {step!r}"
            )
        if step < 1:
            raise ValueError(
                f"step selection {steps!r}: steps are 1-based and positive, "
                f"not {int(step)}"
            )
    steps = tuple(int(step) for step in steps)
    if len(set(steps)) != len(steps):
        raise ValueError(f"step selection {steps!r} names a step twice")
    return steps


def select_steps(
    tensor: Tensor, steps: Optional[Tuple[int, ...]], dim: int = 1
) -> Tensor:
    """``tensor`` at the 1-based ``steps`` along ``dim``; ``None`` is the tensor.

    A leading run ``(1, ..., N)`` narrows (a view of the first ``N`` slots);
    any other selection gathers with ``index_select``.
    """
    if steps is None:
        return tensor
    count = len(steps)
    if steps == tuple(range(1, count + 1)):
        return tensor.narrow(dim, 0, count)
    indices = torch.tensor(
        [step - 1 for step in steps], device=tensor.device, dtype=torch.long
    )
    return tensor.index_select(dim, indices)


def step_ages(steps: Tuple[int, ...], dt: float, time_origin: str, device) -> Tensor:
    """The age in seconds of each selected slot, float32 on ``device``.

    ``time_origin="current"``: slot ``k`` is ``k * dt`` old. ``time_origin=
    "first_slot"``: slot ``k`` is ``(k - 1) * dt`` old, the first stored slot
    having age zero. Each reproduces the formula its kernels carried bit for bit.
    """
    if time_origin not in _TIME_ORIGINS:
        raise ValueError(
            f"time_origin must be one of {sorted(_TIME_ORIGINS)}, not {time_origin!r}"
        )
    t = torch.as_tensor(steps, dtype=torch.float32, device=device)
    return (t - _TIME_ORIGINS[time_origin]) * dt


class BoundKernel(Protocol):
    """What :func:`required_history_depth` reads of a bound component: the two
    ``MdpComponent`` methods, named here because this module cannot import
    ``mdp_component`` (which imports it)."""

    def step_selections(self) -> Dict[str, Optional[Tuple[int, ...]]]: ...

    def get_bindings_dict(self) -> Dict[str, str]: ...


def required_history_depth(
    components: Mapping[str, BoundKernel],
    first_lag: Callable[[str], Optional[int]],
    single_slot_roots: Collection[str] = (),
) -> int:
    """How many past slots the bound components need from the history ring.

    ``first_lag(root)`` is the lag of slot 0 under a context root the ring
    publishes (``None`` for a root that is not the ring's). A component
    reading such a root through a step selection needs ``first_lag(root) +
    max(steps) - 1`` slots; a root in ``single_slot_roots`` (the one-slot
    names such as ``previous_action``) needs ``first_lag(root)`` and no
    selection; any other ring root bound by a component with no step
    selection is refused, naming the component and the argument. ``0`` when
    nothing reads past the current state.
    """
    depth = 0
    for name, component in components.items():
        selections = [
            steps for steps in component.step_selections().values() if steps is not None
        ]
        reach = max((max(steps) for steps in selections), default=None)
        for argument, path in component.get_bindings_dict().items():
            root = path.split(".", 1)[0]
            lag = first_lag(root)
            if lag is None:
                continue
            if root in single_slot_roots:
                depth = max(depth, lag)
                continue
            if reach is None:
                raise ValueError(
                    f"{name}, argument '{argument}': reads the history name "
                    f"'{path}' but declares no step selection; give the kernel a "
                    "history_steps: StepSelection parameter"
                )
            depth = max(depth, lag + reach - 1)
    return depth


__all__ = [
    "StepSelection",
    "WINDOW_PREFIXES",
    "is_window_path",
    "normalize_steps",
    "required_history_depth",
    "select_steps",
    "step_ages",
]
