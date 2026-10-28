# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare a graph-safe Metatron candidate with the untouched pinned ONNX."""

import argparse
import json
from pathlib import Path
import struct

import numpy as np
import onnxruntime as ort


def samples_from_fixture(path: Path, manifest: dict):
    with path.open("rb") as stream:
        if stream.read(12) != b"PM_SCENE_V1\0":
            raise ValueError("unexpected native fixture header")
        count = struct.unpack("<I", stream.read(4))[0]
        if not 1 <= count <= 1000:
            raise ValueError("invalid fixture sample count")
        for _ in range(count):
            feed = {}
            for name, shape, kind in zip(
                manifest["input_names"],
                manifest["input_shapes"],
                manifest["input_dtypes"],
                strict=True,
            ):
                elements = int(np.prod(shape, dtype=np.int64))
                dtype = np.dtype("u1" if kind == "bool" else "<f4")
                data = stream.read(elements * dtype.itemsize)
                if len(data) != elements * dtype.itemsize:
                    raise ValueError("truncated native fixture input")
                feed[name] = (
                    np.frombuffer(data, dtype=dtype).copy().reshape((1, *shape))
                )
                if kind == "bool":
                    feed[name] = feed[name].astype(np.bool_)
            reference = np.frombuffer(stream.read(4 * 29 * 4), dtype="<f4")
            if reference.size != 4 * 29:
                raise ValueError("truncated native fixture output")
            yield feed, reference.reshape(4, 29)


def validate(
    original: Path,
    candidate: Path,
    manifest_path: Path,
    fixture: Path,
    tolerance: float,
) -> None:
    manifest = json.loads(manifest_path.read_text())
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    reference_session = ort.InferenceSession(
        str(original), options, providers=["CPUExecutionProvider"]
    )
    candidate_session = ort.InferenceSession(
        str(candidate), options, providers=["CPUExecutionProvider"]
    )
    maxima = np.zeros(4, dtype=np.float64)
    offset_error = 0.0
    scale = np.asarray(manifest["action_map"]["scale"], np.float32)
    offset = np.asarray(manifest["action_map"]["offset"], np.float32)
    count = 0
    for feed, recorded in samples_from_fixture(fixture, manifest):
        original_outputs = reference_session.run(None, feed)
        candidate_outputs = candidate_session.run(None, feed)
        for index, (reference, actual) in enumerate(
            zip(original_outputs, candidate_outputs, strict=True)
        ):
            maxima[index] = max(
                maxima[index], float(np.max(np.abs(reference - actual)))
            )
            if not np.all(np.isfinite(actual)):
                raise ValueError(f"non-finite candidate output {index}")
        action = candidate_outputs[0][0]
        raw = candidate_outputs[3][0]
        offset_error = max(
            offset_error, float(np.max(np.abs(action - (offset + scale * raw))))
        )
        count += 1
    print(
        f"graphsafe_parity samples={count} max_abs={maxima.tolist()} "
        f"action_affine_max_abs={offset_error}"
    )
    if count == 0 or max(maxima) > tolerance or offset_error > tolerance:
        raise SystemExit("graph-safe candidate failed parity qualification")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    validate(args.original, args.candidate, args.manifest, args.fixture, args.tolerance)


if __name__ == "__main__":
    main()
