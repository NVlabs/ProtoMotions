# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment history sampled at policy rate, before the next action is applied."""

from __future__ import annotations

from collections import deque
from typing import Mapping, Sequence

import numpy as np

from deployment.contract import HistorySlotSpec, InputSpec, ObservationHistorySpec
from deployment.tracker_inputs import (
    ACTION_HISTORY_INPUTS,
    action_history_fill,
    resolve_fill,
)
from protomotions.envs.context_sources import (
    HISTORICAL_NAMESPACES,
    SENSOR_SOURCES,
    current_namespace,
    source_of_path,
)


def history_specs(
    path: str,
    shape: Sequence[int],
    pd_action_offset: Sequence[float] | None = None,
) -> tuple[HistorySlotSpec | None, ObservationHistorySpec | None]:
    """Describe incoming commands and past measurements, using environment timing.

    At state s(t), current_action is a(t-1). Historical slot k starts at k=1
    and holds s(t-k) with a(t-k-1). Actions reset to zero; measurements repeat
    the reset reading. Shape includes the batch dimension.
    """
    if path in ACTION_HISTORY_INPUTS:
        output, lag = ACTION_HISTORY_INPUTS[path]
    elif path.split(".")[0] in HISTORICAL_NAMESPACES:
        leaf = path.split(".", 1)[1]
        if leaf not in ("raw_actions", "processed_actions"):
            source = source_of_path(path)
            if source not in SENSOR_SOURCES:
                return None, None
            current_key = f"{current_namespace(source)}.{leaf}"
            return None, ObservationHistorySpec(
                current_key, tuple(range(1, shape[1] + 1))
            )
        output = "actions" if leaf == "raw_actions" else "joint_pos_targets"
        lag = 2
    else:
        return None, None
    slots = shape[1] if len(shape) == 3 else 1
    fill_key = (
        "raw_action_history" if output == "actions" else "processed_action_history"
    )
    fill = action_history_fill(fill_key, pd_action_offset)
    return HistorySlotSpec(
        output,
        slots,
        lag,
        fill,
        tuple(range(lag, lag + slots)),
        0.0 if fill == "zeros" else None,
    ), None


class InputHistory:
    """Replay the history declared by a contract using NumPy arrays.

    Call reset(first_measurements), then inputs() before inference. After the
    command is accepted, commit the pre-action measurements and the *applied*
    raw action/PD targets. Never record a rejected or unmodified network action
    when the controller actually applied something else.
    """

    def __init__(self, specs: Sequence[InputSpec]) -> None:
        self.specs = tuple(
            spec for spec in specs if spec.history or spec.observation_history
        )
        self._values: dict[str, deque[np.ndarray]] = {}

    def reset(self, measurements: Mapping[str, np.ndarray]) -> None:
        self._values.clear()
        for spec in self.specs:
            if spec.history is not None:
                key = spec.history.output_key
                depth = max(spec.history.steps_back)
                initial = np.broadcast_to(
                    resolve_fill(spec.history.reset_fill, spec.shape[-1]),
                    (spec.shape[0], spec.shape[-1]),
                ).astype(spec.dtype)
            else:
                key = spec.observation_history.current_key
                depth = max(spec.observation_history.steps_back)
                initial = np.asarray(measurements[key], dtype=spec.dtype)
                expected = (spec.shape[0], *spec.shape[2:])
                if initial.shape != expected:
                    raise ValueError(f"{key}: expected {expected}, got {initial.shape}")
            depth = max(depth, len(self._values.get(key, ())))
            self._values[key] = deque(
                (initial.copy() for _ in range(depth)), maxlen=depth
            )

    def inputs(self) -> dict[str, np.ndarray]:
        result = {}
        for spec in self.specs:
            history = spec.history or spec.observation_history
            key = history.output_key if spec.history else history.current_key
            values = self._values[key]
            result[spec.key] = (
                np.stack([values[step - 1] for step in history.steps_back], axis=1)
                .reshape(spec.shape)
                .copy()
            )
        return result

    def commit(
        self,
        before_step: Mapping[str, np.ndarray],
        applied_outputs: Mapping[str, np.ndarray],
    ) -> None:
        for key, values in self._values.items():
            source = (
                applied_outputs
                if key in ("actions", "joint_pos_targets")
                else before_step
            )
            value = np.asarray(source[key], dtype=values[0].dtype)
            if value.shape != values[0].shape:
                raise ValueError(
                    f"{key}: expected {values[0].shape}, got {value.shape}"
                )
            values.appendleft(value.copy())
