# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the pinned G1 scene-policy FK as a small graph-safe ONNX model.

Run once using the matching ProtoMotions source. The resulting graph takes
only sensor-available root/joint state; MuJoCo and Unitree will use the same
native TensorRT engine. This script never runs in the control loop.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from protomotions.components.pose_lib import (
    compute_forward_kinematics_from_transforms,
    extract_kinematic_info,
    extract_transforms_from_qpos,
    fk_batch_mjcf_with_velocities,
)


def _matrix_to_quaternion_graphsafe(matrix: torch.Tensor) -> torch.Tensor:
    """Value-equivalent ProtoMotions conversion with fixed-shape selection."""
    batch = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch + (9,)), dim=-1
    )
    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                [
                    1 + m00 + m11 + m22,
                    1 + m00 - m11 - m22,
                    1 - m00 + m11 - m22,
                    1 - m00 - m11 + m22,
                ],
                dim=-1,
            ),
            min=0,
        )
    )
    candidates = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    ) / (2 * q_abs[..., None].clamp(min=0.1))
    index = q_abs.argmax(dim=-1)[..., None, None].expand(*batch, 1, 4)
    wxyz = torch.gather(candidates, -2, index).squeeze(-2)
    return torch.cat((wxyz[..., 1:], wxyz[..., :1]), dim=-1)


class SceneFK(torch.nn.Module):
    def __init__(self, kinematic_info):
        super().__init__()
        self.info = kinematic_info

    def forward(self, root_pos, root_rot_xyzw, joints):
        root_rot_wxyz = torch.cat((root_rot_xyzw[:, 3:], root_rot_xyzw[:, :3]), dim=-1)
        qpos = torch.cat((root_pos, root_rot_wxyz, joints), dim=-1)
        pos, matrices = compute_forward_kinematics_from_transforms(
            self.info, *extract_transforms_from_qpos(self.info, qpos)
        )
        return pos, _matrix_to_quaternion_graphsafe(matrices)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["contract"] != "protomotions.masked_mimic.g1.v5_raw_context_scene":
        raise ValueError("wrong policy contract for G1 FK")
    info = extract_kinematic_info(str(args.xml))
    if list(info.body_names) != manifest["robot_contract"]["body_names"]:
        raise ValueError("MJCF body order differs from the pinned scene policy")
    if info.nq != 36:
        raise ValueError("G1 scene FK expects 7 free-root and 29 hinge coordinates")
    model = SceneFK(info).eval()
    root = torch.tensor([[0.0, 0.0, 0.75]], dtype=torch.float32)
    rotation = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    joints = torch.tensor(manifest["action_map"]["offset"], dtype=torch.float32)[None]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite FK artifact: {args.output}")
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                (root, rotation, joints),
                str(partial),
                opset_version=18,
                input_names=["root_pos", "root_rot", "dof_pos"],
                output_names=["body_pos", "body_rot"],
                dynamo=False,
            )
        exported = onnx.load(str(partial))
        onnx.checker.check_model(exported)
        if any(
            node.op_type in {"NonZero", "If", "Loop", "Scan"}
            for node in exported.graph.node
        ):
            raise ValueError("FK export contains a data-dependent shape/control op")
        session = ort.InferenceSession(str(partial), providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(17)
        offset = np.asarray(manifest["action_map"]["offset"], np.float32)
        max_pos = max_rot = 0.0
        for _ in range(20):
            sample_pos = rng.uniform([-0.3, -0.3, 0.4], [0.3, 0.3, 0.9], (1, 3)).astype(
                np.float32
            )
            sample_rot = rng.normal(size=(1, 4)).astype(np.float32)
            sample_rot /= np.linalg.norm(sample_rot, axis=-1, keepdims=True)
            sample_joints = (offset + rng.uniform(-0.25, 0.25, 29)).astype(np.float32)[
                None
            ]
            out_pos, out_rot = session.run(
                None,
                {
                    "root_pos": sample_pos,
                    "root_rot": sample_rot,
                    "dof_pos": sample_joints,
                },
            )
            qpos = torch.from_numpy(
                np.concatenate(
                    (
                        sample_pos,
                        sample_rot[:, 3:],
                        sample_rot[:, :3],
                        sample_joints,
                    ),
                    axis=-1,
                )
            )
            with torch.no_grad():
                reference = fk_batch_mjcf_with_velocities(
                    info, qpos, compute_velocities=False
                )
            ref_pos = reference.rigid_body_pos.numpy()
            ref_rot = reference.rigid_body_rot.numpy()
            max_pos = max(max_pos, float(np.max(np.abs(ref_pos - out_pos))))
            max_rot = max(
                max_rot,
                float(
                    np.max(
                        np.minimum(
                            np.abs(ref_rot - out_rot),
                            np.abs(ref_rot + out_rot),
                        )
                    )
                ),
            )
        if max_pos > 1e-5 or max_rot > 1e-5:
            raise ValueError(
                f"FK ONNX differs from ProtoMotions: {max_pos=}, {max_rot=}"
            )
        partial.replace(args.output)
        print(
            f"FK ONNX ready {args.output} nodes={len(exported.graph.node)} "
            f"max_pos_m={max_pos:.8g} max_quat_component={max_rot:.8g}",
            flush=True,
        )
    finally:
        partial.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
