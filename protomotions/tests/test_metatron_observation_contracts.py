# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metatron body-pose declarations and batch-independent anchor masking."""

import pytest
import torch

from protomotions.envs.context_views import deploy_for
from protomotions.envs.obs.masked_mimic import compute_target_poses_only


@pytest.mark.parametrize("source", ["clean", "noisy"])
def test_sensor_body_pose_has_explicit_deploy_kinds(source):
    assert deploy_for(f"{source}.rigid_body_pos") == "body_pos"
    assert deploy_for(f"{source}.rigid_body_rot") == "body_rot"


class _AnchorGoals(torch.nn.Module):
    def __init__(self, include_root_relative):
        super().__init__()
        self.include_root_relative = include_root_relative
        self.register_buffer("body_ids", torch.tensor([2, 0]))

    def forward(self, pos, rot, target_pos, target_rot, masks, translation, rotation):
        return compute_target_poses_only(
            self.body_ids,
            pos,
            rot,
            target_pos,
            target_rot,
            masks,
            anchor_body_index=2,
            include_root_relative=self.include_root_relative,
            anchor_translation_mask=translation,
            anchor_rotation_mask=rotation,
        )


def _goal_inputs(batch_size):
    generator = torch.Generator().manual_seed(38)
    pos = torch.randn(batch_size, 3, 3, generator=generator)
    rot = torch.nn.functional.normalize(
        torch.randn(batch_size, 3, 4, generator=generator), dim=-1
    )
    return {
        "pos": pos,
        "rot": rot,
        "target_pos": torch.randn(batch_size, 4, 3, 3, generator=generator),
        "target_rot": torch.nn.functional.normalize(
            torch.randn(batch_size, 4, 3, 4, generator=generator), dim=-1
        ),
        "masks": torch.rand(batch_size, 4 * 2 * 2, generator=generator) > 0.5,
        "translation": torch.rand(batch_size, 4, 2, generator=generator) > 0.5,
        "rotation": torch.rand(batch_size, 4, 2, generator=generator) > 0.5,
    }


@pytest.mark.parametrize("include_root_relative", [False, True])
def test_anchor_mask_export_preserves_dynamic_batch(tmp_path, include_root_relative):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    model = _AnchorGoals(include_root_relative).eval()
    inputs = _goal_inputs(1)
    path = tmp_path / "anchor-goals.onnx"
    torch.onnx.export(
        model,
        tuple(inputs.values()),
        path,
        input_names=list(inputs),
        output_names=["goals"],
        dynamic_axes={name: {0: "batch"} for name in [*inputs, "goals"]},
        opset_version=17,
        dynamo=False,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for batch_size in (1, 3):
        for visibility in ("mixed", "all", "none"):
            inputs = _goal_inputs(batch_size)
            if visibility != "mixed":
                for name in ("masks", "translation", "rotation"):
                    inputs[name].fill_(visibility == "all")
            with torch.no_grad():
                expected = model(*inputs.values())
            (actual,) = session.run(
                None, {name: value.numpy() for name, value in inputs.items()}
            )
            torch.testing.assert_close(torch.from_numpy(actual), expected)
