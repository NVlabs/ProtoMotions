# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One history for physical state, sensor estimates, and incoming commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Mapping, Optional, Tuple

import torch
from torch import Tensor

from protomotions.envs.context_sources import HISTORICAL_NAMESPACES
from protomotions.envs.state import BodyState, StateSource
from protomotions.envs.steps import BoundKernel, required_history_depth


@dataclass
class HistorySample:
    """States and commands sharing the same environment and time dimensions.

    Raw actions are policy outputs; processed actions are simulator commands.
    An action is stored with the state reached by applying it. The commands are
    shared across sources, never copied into each source's state record.
    """

    states: Mapping[StateSource, BodyState]
    raw_actions: Tensor
    processed_actions: Tensor

    def map(self, transform: Callable[[Tensor], Tensor]) -> HistorySample:
        """Transform every available tensor, preserving the source schema."""
        return HistorySample(
            states={key: state.map(transform) for key, state in self.states.items()},
            raw_actions=transform(self.raw_actions),
            processed_actions=transform(self.processed_actions),
        )

    def tensors(self) -> Iterator[tuple[str, Tensor]]:
        """Yield named tensors for shape/schema checks and in-place writes."""
        for source, state in self.states.items():
            for name, value in state.tensors():
                yield f"{source}.{name}", value
        yield "raw_actions", self.raw_actions
        yield "processed_actions", self.processed_actions


# The context roots the ring publishes as one past slot each.
SINGLE_SLOT_ROOTS: Tuple[str, ...] = ("previous_action", "previous_processed_action")


def first_lag(root: str) -> Optional[int]:
    """Lag of slot 0 under a context root the ring publishes, None for any
    other root: the historical namespaces start one step back, the two
    action rings start at the current command (slot 0, lag 0), and the two
    ``previous_*`` names are one past slot."""
    if root in HISTORICAL_NAMESPACES or root in SINGLE_SLOT_ROOTS:
        return 1
    if root in ("raw_action_history", "processed_action_history"):
        return 0
    return None


def history_depth(components: Mapping[str, BoundKernel]) -> int:
    """How many past slots the ring keeps for ``components``: the one
    computation (:func:`protomotions.envs.steps.required_history_depth`) over
    the ring's roots. 0 when no component reads the past; the historical
    namespaces and the ``previous_*`` names are then not published, which is
    consistent by construction (a binding to one of them is what makes the
    depth positive). A component that reads a history window without a step
    selection is refused here, by name."""
    return required_history_depth(components, first_lag, SINGLE_SLOT_ROOTS)


class StateHistoryBuffer:
    """Store current state at slot 0 and previous states at slots 1 through H.

    Construction and reset repeat the measured post-reset state into all slots,
    including the noisy estimate, and zero incoming commands. No reference-clip
    poses or privileged substitutes are inserted into sensor history.

    The initial sample defines available sources and fields. Subsequent writes
    must have the same schema; adding a sensor channel does not require editing
    this class. ``capture`` gives the context its own snapshot so later resets
    and writes cannot change observations retained by a caller.
    """

    # The two slot facts a deployment needs to replay the action rings (the
    # exporter states them next to each action-history input): slot 0 holds
    # the action that produced the current state, one policy call back, and
    # a reset fills the rings with zeros.
    ACTION_LAG = 1
    RESET_FILL = "zeros"

    def __init__(self, num_history_steps: int, initial: HistorySample) -> None:
        if num_history_steps < 0:
            raise ValueError("num_history_steps must be nonnegative")
        self.num_history_steps = num_history_steps
        self._values = initial.map(
            lambda value: value.unsqueeze(1)
            .expand(-1, num_history_steps + 1, *value.shape[1:])
            .clone()
        )
        self._values.raw_actions.zero_()
        self._values.processed_actions.zero_()

    @torch.no_grad()
    def push(self, sample: HistorySample) -> None:
        """Advance all sources once and insert the resulting state/commands."""
        pairs = self._matching_tensors(sample, history=False)
        for target, value in pairs:
            target[:, 1:] = target[:, :-1].clone()
            target[:, 0] = value

    @torch.no_grad()
    def reset(self, env_ids: Tensor, sample: HistorySample) -> None:
        """Repeat subset post-reset states and clear only their commands."""
        pairs = self._matching_tensors(sample, history=False, num_envs=len(env_ids))
        for target, value in pairs:
            target[env_ids] = value.unsqueeze(1)
        self._values.raw_actions[env_ids] = 0
        self._values.processed_actions[env_ids] = 0

    def capture(self, env_ids: Optional[Tensor] = None) -> HistorySample:
        """Return one owned snapshot; current/history views slice it directly."""
        if env_ids is None:
            return self._values.map(torch.clone)
        # Tensor indexing already makes an owned copy of the selected rows.
        return self._values.map(lambda value: value[env_ids])

    def save_state(self) -> HistorySample:
        """Snapshot the complete history without sampling or reconstruction."""
        return self.capture()

    @torch.no_grad()
    def load_state(self, snapshot: HistorySample) -> None:
        """Restore values exactly; schema mismatches are errors, not migrations."""
        for target, value in self._matching_tensors(snapshot, history=True):
            target.copy_(value)

    def _matching_tensors(
        self,
        sample: HistorySample,
        *,
        history: bool,
        num_envs: Optional[int] = None,
    ) -> list[tuple[Tensor, Tensor]]:
        # Validate the whole write before changing storage, including partial
        # resets, so an invalid sample cannot leave mixed old/new state behind.
        targets = dict(self._values.tensors())
        values = dict(sample.tensors())
        if targets.keys() != values.keys():
            raise ValueError("History sample sources/fields differ from initial schema")
        pairs = []
        for name, target in targets.items():
            value = values[name]
            shape = target.shape if history else target[:, 0].shape
            if num_envs is not None:
                shape = (num_envs, *shape[1:])
            if (
                value.shape != shape
                or value.dtype != target.dtype
                or value.device != target.device
            ):
                raise ValueError(
                    f"History tensor {name} has incompatible shape, dtype, or device"
                )
            pairs.append((target, value))
        return pairs
