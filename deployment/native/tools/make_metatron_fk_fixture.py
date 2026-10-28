# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write deterministic ONNX FK references for native TensorRT parity tests."""

import argparse
import json
from pathlib import Path
import struct

import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.samples <= 1000:
        raise ValueError("samples must be between 1 and 1000")
    manifest = json.loads(args.manifest.read_text())
    offset = np.asarray(manifest["action_map"]["offset"], np.float32)
    if offset.shape != (29,):
        raise ValueError("expected 29 joint offsets")
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(17)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        output.write(b"PM_FK_V1\0\0\0\0")
        output.write(struct.pack("<I", args.samples))
        for sample in range(args.samples):
            root_pos = rng.uniform((-0.3, -0.3, 0.4), (0.3, 0.3, 0.9)).astype(
                np.float32
            )[None]
            root_rot = rng.normal(size=(1, 4)).astype(np.float32)
            root_rot /= np.linalg.norm(root_rot, axis=-1, keepdims=True)
            joints = (offset + rng.uniform(-0.25, 0.25, 29)).astype(np.float32)[None]
            inputs = {"root_pos": root_pos, "root_rot": root_rot, "dof_pos": joints}
            body_pos, body_rot = session.run(None, inputs)
            for values in (root_pos, root_rot, joints, body_pos, body_rot):
                output.write(np.ascontiguousarray(values, dtype="<f4").tobytes())
    print(f"FK fixture ready {args.output} samples={args.samples}")


if __name__ == "__main__":
    main()
