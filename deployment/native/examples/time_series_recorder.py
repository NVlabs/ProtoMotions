# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, background time-series recording for deployment qualification."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

import msgpack

try:
    import msgpack_numpy
except ImportError:  # pragma: no cover - the deployment environment installs it
    msgpack_numpy = None

import numpy as np

if msgpack_numpy is not None:
    msgpack_numpy.patch()


def iter_records(path: str | os.PathLike[str]):
    """Stream deployment records from a run directory or ``records.msgpack``.

    Records are ordinary dictionaries containing NumPy arrays. Streaming keeps
    long qualification runs analyzable without loading the whole trajectory.
    """
    records_path = Path(path).expanduser()
    if records_path.is_dir():
        records_path /= "records.msgpack"
    with records_path.open("rb") as stream:
        yield from msgpack.Unpacker(stream, raw=False)


def load_records(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load all records for short runs; use :func:`iter_records` for long runs."""
    return list(iter_records(path))


def _snapshot(value: Any) -> Any:
    """Copy mutable array/list data before it crosses the control thread."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_snapshot(item) for item in value)
    return value


class TimeSeriesRecorder:
    """Write replayable records without performing disk I/O in inference.

    ``append`` only snapshots arrays and performs a non-blocking queue put.
    Serialization, compression-independent msgpack framing, and all file I/O
    happen on the writer thread. If the writer cannot keep up, records are
    dropped and counted rather than delaying the policy or actuator command.
    """

    def __init__(
        self,
        root: str | os.PathLike[str] = "logs",
        metadata: dict[str, Any] | None = None,
        queue_size: int = 256,
    ) -> None:
        if queue_size < 1:
            raise ValueError("recorder queue_size must be positive")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = Path(root).expanduser() / f"run_{timestamp}_{os.getpid()}"
        self.path.mkdir(parents=True, exist_ok=True)
        self.records_path = self.path / "records.msgpack"
        self.metadata_path = self.path / "metadata.json"
        self._metadata = dict(metadata or {})
        self._metadata.update(
            {
                "schema": "protomotions.deployment.timeseries.v1",
                "record_format": "concatenated-msgpack",
                "created_wall_time_ns": time.time_ns(),
                "queue_size": int(queue_size),
            }
        )
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(
            maxsize=queue_size
        )
        self._stop = threading.Event()
        self._closed = False
        self._written = 0
        self._dropped = 0
        self._thread = threading.Thread(
            target=self._writer, name="deployment-timeseries-writer", daemon=True
        )
        self._write_metadata()
        self._thread.start()

    def _write_metadata(self) -> None:
        self.metadata_path.write_text(
            json.dumps(self._metadata, indent=2, default=str) + "\n"
        )

    def append(self, record: dict[str, Any]) -> bool:
        if self._closed:
            return False
        item = _snapshot(record)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self._dropped += 1
            return False
        return True

    def _writer(self) -> None:
        with self.records_path.open("ab") as stream:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    self._queue.task_done()
                    continue
                payload = msgpack.packb(item, use_bin_type=True)
                stream.write(payload)
                self._written += 1
                if self._written % 50 == 0:
                    stream.flush()
                self._queue.task_done()
            stream.flush()
            os.fsync(stream.fileno())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._thread.join(timeout=10.0)
        self._metadata.update(
            {
                "records_written": int(self._written),
                "records_dropped": int(self._dropped),
                "closed_wall_time_ns": time.time_ns(),
            }
        )
        self._write_metadata()

    @property
    def records_written(self) -> int:
        return self._written

    @property
    def records_dropped(self) -> int:
        return self._dropped

    def __enter__(self) -> "TimeSeriesRecorder":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
