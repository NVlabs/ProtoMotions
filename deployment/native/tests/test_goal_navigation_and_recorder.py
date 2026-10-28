# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import msgpack_numpy
import numpy as np

from examples.goal_navigation import RelativeGoalNavigator
from examples.latency_profiler import LatencyProfiler
from examples.time_series_recorder import TimeSeriesRecorder, load_records


class GoalNavigationAndRecorderTest(unittest.TestCase):
    def test_goal_is_world_frame_and_arrival_is_one_shot(self) -> None:
        navigator = RelativeGoalNavigator()
        goal = navigator.start(np.zeros(3), np.pi / 2.0, 0.0)
        np.testing.assert_allclose(goal, [0.0, 2.0, 0.0], atol=1e-7)
        self.assertFalse(navigator.update(np.asarray([1.0, 1.0, 0.0])))
        self.assertTrue(navigator.update(np.asarray([0.0, 1.81, 0.0])))
        self.assertFalse(navigator.update(np.asarray([0.0, 1.81, 0.0])))

    def test_recorder_writes_replayable_records_off_thread(self) -> None:
        msgpack_numpy.patch()
        with tempfile.TemporaryDirectory() as directory:
            recorder = TimeSeriesRecorder(directory, queue_size=8)
            source = np.arange(4, dtype=np.float32)
            self.assertTrue(recorder.append({"t": 1, "state": source}))
            source[:] = -1.0
            recorder.close()
            self.assertEqual(recorder.records_written, 1)
            self.assertEqual(recorder.records_dropped, 0)
            metadata = json.loads((recorder.path / "metadata.json").read_text())
            self.assertEqual(metadata["records_written"], 1)
            records = load_records(recorder.path)
            self.assertEqual(len(records), 1)
            np.testing.assert_allclose(records[0]["state"], [0, 1, 2, 3])

    def test_latency_profiler_reports_stage_percentiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "latency.json"
            profiler = LatencyProfiler(output, samples=2)
            for offset in (0, 10):
                profiler.record(
                    cycle_start_ns=offset,
                    sdk_state_ready_ns=offset + 100,
                    sensor_ready_ns=offset + 200,
                    operator_ready_ns=offset + 300,
                    policy_ready_ns=offset + 500,
                    action_ready_ns=offset + 550,
                    sdk_send_done_ns=offset + 600,
                )
            result = profiler.close({"backend": "test"})
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["samples"], 2)
            self.assertEqual(result["metadata"]["backend"], "test")
            self.assertAlmostEqual(
                result["latency_us"]["sdk_state_to_send"]["p50_us"],
                0.5,
            )


if __name__ == "__main__":
    unittest.main()
