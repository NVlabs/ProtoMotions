# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in control-loop latency profiling for deployment qualification."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Mapping

import numpy as np


_STAGES = (
    "sdk_read",
    "sensor_contract",
    "operator_and_goal",
    "policy",
    "action_guard",
    "sdk_send",
    "sdk_state_to_send",
    "total_cycle",
)


class LatencyProfiler:
    """Collect fixed-size timing vectors without disk I/O on the control thread.

    The caller supplies monotonic timestamps around the real loop boundaries.
    Only integer subtraction and list appends occur while profiling; JSON
    serialization happens when :meth:`close` is called after the loop exits.
    """

    def __init__(self, output: str | Path, samples: int) -> None:
        if samples < 1:
            raise ValueError("latency profiler sample count must be positive")
        self.output = Path(output).expanduser()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.requested_samples = int(samples)
        self._durations: dict[str, list[int]] = {stage: [] for stage in _STAGES}
        self.cycles_seen = 0
        self.failures = 0
        self.started_wall_time_ns = time.time_ns()

    @staticmethod
    def now_ns() -> int:
        return time.perf_counter_ns()

    @property
    def complete(self) -> bool:
        return len(self._durations["total_cycle"]) >= self.requested_samples

    @property
    def samples(self) -> int:
        return len(self._durations["total_cycle"])

    def record(
        self,
        *,
        cycle_start_ns: int,
        sdk_state_ready_ns: int,
        sensor_ready_ns: int,
        operator_ready_ns: int,
        policy_ready_ns: int,
        action_ready_ns: int,
        sdk_send_done_ns: int,
    ) -> None:
        """Record one successful SDK-state-to-command cycle."""

        self.cycles_seen += 1
        if self.complete:
            return
        points = (
            cycle_start_ns,
            sdk_state_ready_ns,
            sensor_ready_ns,
            operator_ready_ns,
            policy_ready_ns,
            action_ready_ns,
            sdk_send_done_ns,
        )
        if any(later < earlier for earlier, later in zip(points, points[1:])):
            self.failures += 1
            return
        durations = {
            "sdk_read": sdk_state_ready_ns - cycle_start_ns,
            "sensor_contract": sensor_ready_ns - sdk_state_ready_ns,
            "operator_and_goal": operator_ready_ns - sensor_ready_ns,
            "policy": policy_ready_ns - operator_ready_ns,
            "action_guard": action_ready_ns - policy_ready_ns,
            "sdk_send": sdk_send_done_ns - action_ready_ns,
            "sdk_state_to_send": sdk_send_done_ns - sdk_state_ready_ns,
            "total_cycle": sdk_send_done_ns - cycle_start_ns,
        }
        for stage, duration in durations.items():
            self._durations[stage].append(int(duration))

    def _summary(self, values: list[int]) -> dict[str, float | int]:
        if not values:
            return {"samples": 0}
        array = np.asarray(values, dtype=np.float64) / 1_000.0
        return {
            "samples": len(values),
            "mean_us": float(np.mean(array)),
            "p50_us": float(np.percentile(array, 50.0)),
            "p95_us": float(np.percentile(array, 95.0)),
            "p99_us": float(np.percentile(array, 99.0)),
            "min_us": float(np.min(array)),
            "max_us": float(np.max(array)),
        }

    def close(self, metadata: Mapping[str, object] | None = None) -> dict[str, object]:
        """Write the profile JSON and return the serialized result."""

        result: dict[str, object] = {
            "status": "ok"
            if self.samples == self.requested_samples and self.failures == 0
            else "incomplete",
            "record": "protomotions.deployment.end_to_end_latency",
            "host": platform.node(),
            "boundary": "Unitree SDK state acquisition -> Unitree SDK step() return",
            "scope": (
                "real deployment control thread; SDK call boundaries include Python binding overhead; "
                "DDS wire delivery after step() return is not measured"
            ),
            "requested_samples": self.requested_samples,
            "samples": self.samples,
            "cycles_seen": self.cycles_seen,
            "failures": self.failures,
            "started_wall_time_ns": self.started_wall_time_ns,
            "closed_wall_time_ns": time.time_ns(),
            "latency_us": {
                stage: self._summary(values)
                for stage, values in self._durations.items()
            },
        }
        if metadata:
            result["metadata"] = dict(metadata)
        self.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result
