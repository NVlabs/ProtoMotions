# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert a ProtoMotions packaged motion library to native PMMOT4."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

_MAGIC = b"PMMOT4\0\0"
_VERSION = 1


def _array(value: object, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def convert(source: Path, destination: Path) -> None:
    data = torch.load(source, map_location="cpu", weights_only=False)
    required = ("length_starts", "motion_num_frames", "motion_dt", "gts", "grs")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"packaged motion library is missing keys: {missing}")
    starts = _array(data["length_starts"], "length_starts").astype(np.int64)
    frames = _array(data["motion_num_frames"], "motion_num_frames").astype(np.int64)
    dts = _array(data["motion_dt"], "motion_dt").astype(np.float32)
    positions = _array(data["gts"], "gts").astype(np.float32)
    rotations = _array(data["grs"], "grs").astype(np.float32)
    if starts.ndim != 1 or frames.ndim != 1 or dts.ndim != 1:
        raise ValueError("motion index arrays must be one-dimensional")
    if not (len(starts) == len(frames) == len(dts) > 0):
        raise ValueError("motion index arrays have inconsistent lengths")
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError(f"gts must be [frames,bodies,3], got {positions.shape}")
    if rotations.shape != (positions.shape[0], positions.shape[1], 4):
        raise ValueError(f"grs must be [frames,bodies,4], got {rotations.shape}")
    if int(positions.shape[0]) != int(starts[-1] + frames[-1]):
        raise ValueError("motion index does not cover the frame payload")
    if np.any(starts < 0) or np.any(frames <= 0) or np.any(dts <= 0):
        raise ValueError("motion index contains invalid values")
    if np.any(starts[1:] < starts[:-1] + frames[:-1]):
        raise ValueError("motion clips overlap or are out of order")
    destination.parent.mkdir(parents=True, exist_ok=True)
    dofs = int(data.get("dof_count", 29))
    with destination.open("wb") as stream:
        stream.write(
            struct.pack(
                "<8sIIIIQ",
                _MAGIC,
                _VERSION,
                int(positions.shape[1]),
                dofs,
                len(starts),
                int(positions.shape[0]),
            )
        )
        for start, count, dt in zip(starts, frames, dts):
            stream.write(struct.pack("<QIf", int(start), int(count), float(dt)))
        positions.astype("<f4", copy=False).tofile(stream)
        rotations.astype("<f4", copy=False).tofile(stream)
    print(
        f"wrote {destination}: motions={len(starts)} frames={positions.shape[0]} "
        f"bodies={positions.shape[1]} dofs={dofs}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.source.expanduser(), args.output.expanduser())


if __name__ == "__main__":
    main()
