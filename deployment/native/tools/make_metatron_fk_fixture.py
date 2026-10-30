# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write deterministic ONNX FK references for the native TensorRT parity test.

The exported FK graph (``fk.onnx`` from ``deployment/export_fk_to_onnx.py``) is
run in ONNX Runtime on random joint angles and anchor poses. Each record is
``dof_pos, anchor_pos, anchor_rot, body_pos, body_rot`` as little-endian
float32, after the magic ``PM_FK_V2`` and the sample count; the sidecar gives
the tensor names and the joint count.
"""

import argparse
from pathlib import Path
import struct

import numpy as np
import onnxruntime as ort

from deployment.export_fk_to_onnx import FK_OUTPUTS, FkSidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True, help="fk.onnx")
    parser.add_argument("--sidecar", type=Path, required=True, help="its fk.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.samples <= 1000:
        raise ValueError("samples must be between 1 and 1000")

    sidecar = FkSidecar.from_yaml(args.sidecar)
    # The sidecar's inputs are in the kernel's argument order, which is also the
    # record order: joint angles, anchor position, anchor rotation.
    dof_name, pos_name, rot_name = sidecar.input_names
    num_dofs = sidecar.input_shapes[dof_name][-1]
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(17)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        output.write(b"PM_FK_V2\0\0\0\0")
        output.write(struct.pack("<I", args.samples))
        for _ in range(args.samples):
            dof_pos = rng.uniform(-0.5, 0.5, num_dofs).astype(np.float32)[None]
            anchor_pos = rng.uniform((-0.3, -0.3, 0.5), (0.3, 0.3, 1.0)).astype(
                np.float32
            )[None]
            anchor_rot = rng.normal(size=(1, 4)).astype(np.float32)
            anchor_rot /= np.linalg.norm(anchor_rot, axis=-1, keepdims=True)
            feed = {dof_name: dof_pos, pos_name: anchor_pos, rot_name: anchor_rot}
            body_pos, body_rot = session.run(list(FK_OUTPUTS), feed)
            for values in (dof_pos, anchor_pos, anchor_rot, body_pos, body_rot):
                output.write(np.ascontiguousarray(values, dtype="<f4").tobytes())
    print(f"FK fixture ready {args.output} samples={args.samples}")


if __name__ == "__main__":
    main()
