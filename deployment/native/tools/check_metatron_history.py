# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay a training-export trace through deployment's sensor/action history."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from examples.metatron_v1_contract import (
    IncomingActionHistory,
    SensorReading,
    validate_manifest,
)


def check(
    model: Path, manifest_path: Path, trace_path: Path, provider: str = "cpu"
) -> dict:
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    backend = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}[provider]
    settings = {"use_tf32": False} if provider == "cuda" else {}
    session = ort.InferenceSession(
        str(model), sess_options=options, providers=[(backend, settings)]
    )
    if backend not in session.get_providers():
        raise RuntimeError(f"requested {backend} did not initialize")
    trace = json.loads(trace_path.read_text())
    history = IncomingActionHistory()
    anchor = manifest["robot_contract"]["anchor_body_index"]
    input_errors, max_action_error = {}, 0.0
    for row in trace:
        if row["reset"]:
            history.reset()
        snapshot = row["snapshot"]
        positions = np.asarray(snapshot["bodyPositions"], np.float32).reshape(-1, 3)
        rotations = np.asarray(snapshot["bodyRotations"], np.float32).reshape(-1, 4)
        sensors = SensorReading(
            np.asarray(snapshot["dofPosition"], np.float32),
            np.asarray(snapshot["dofVelocity"], np.float32),
            np.asarray(snapshot["rootLocalAngularVelocity"], np.float32),
            rotations[anchor],
            positions[anchor],
            positions[0],
            np.asarray(snapshot["rootRotation"], np.float32),
        )
        reconstructed = history.inputs(sensors)
        feed = {}
        for key, name, shape, dtype in zip(
            manifest["input_keys"],
            manifest["input_names"],
            manifest["input_shapes"],
            manifest["input_dtypes"],
        ):
            reference = np.asarray(row["inputs"][key], dtype=dtype).reshape(shape)
            value = reconstructed[key] if key in reconstructed else reference
            error = float(np.max(np.abs(value.astype(float) - reference.astype(float))))
            input_errors[key] = max(input_errors.get(key, 0), error)
            np.testing.assert_allclose(value, reference, atol=3e-5, rtol=0, err_msg=key)
            feed[name] = value.reshape(1, *shape)
        raw = session.run(["raw_action"], feed)[0][0]
        error = float(np.max(np.abs(raw - row["rawAction"])))
        max_action_error = max(max_action_error, error)
        np.testing.assert_allclose(raw, row["rawAction"], atol=2e-5, rtol=0)
        history.commit(sensors, np.asarray(row["appliedAction"], np.float32))
    return {
        "passed": True,
        "provider": backend,
        "steps": len(trace),
        "resets": sum(row["reset"] for row in trace),
        "max_raw_action_error": max_action_error,
        "input_errors": input_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.model, args.manifest, args.trace, args.provider)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
