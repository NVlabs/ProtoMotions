# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline acceptance cell for the Metatron MaskedMimic export (item 23, PR 4).

Builds an untrained five-body fixture teacher, builds the Metatron student against
it in memory, and checks the exporter reads it as a MaskedMimic deployable with
the expected observation contract. The full sidecar export stops at the
FK-derived current body positions (a derived-provider design, item 59), so this
asserts the build and the deployable in_keys -- proof the masked build + export
path runs to the observation layer on the real recipe, which the minimal-config
helper tests in test_export_contract_coverage do not cover.
"""

import pytest


def test_metatron_student_is_a_masked_deployable_with_the_expected_obs_keys(
    tmp_path_factory, monkeypatch
):
    import shutil

    pytest.importorskip("mujoco")

    from protomotions.tests.support import recipe_stages as rs

    if not rs.has_payload(rs.MOTION_FILE):
        pytest.skip("motion LFS payload missing")

    from deployment.export_to_onnx import _deployable_policy_in_keys
    from protomotions.agents.common.latent import VAE_LATENT_KEY
    from protomotions.utils.config_builder import build_experiment

    rs.eager_mujoco_build(monkeypatch)

    # The five-body FSQ teacher and the masked-mimic student, under metatron/.
    teacher_dir = tmp_path_factory.mktemp("five_body_teacher")
    teacher = rs.write_checkpoint(
        "examples/experiments/metatron/g1_teacher.py", teacher_dir
    )
    student = build_experiment(
        rs.training_argv(
            "examples/experiments/metatron/g1_student.py",
            rs.MOTION_FILE,
            ["--expert-model-path", str(teacher)],
        )
    )
    # The teacher checkpoint is ~650 MB and is only consumed by build_experiment;
    # drop it now rather than leave it under the pytest tmp tree across runs.
    shutil.rmtree(teacher_dir, ignore_errors=True)

    actor_cfg, in_keys = _deployable_policy_in_keys(student.agent.model)
    assert actor_cfg is None  # a MaskedMimic model has no plain actor
    keys = set(in_keys)

    # The 18 student observation keys: proprioceptive state plus the masked_mimic.*
    # goal/style/velocity conditioning the host supplies.
    expected = {
        "fk_max_coords_obs",
        "reduced_coords_obs",
        "last_action",
        "historical_proprioception",
        "fk_masked_mimic_target_poses",
        "masked_mimic_target_masks",
        "masked_mimic_target_times",
        "masked_mimic_target_poses_masks",
        "masked_mimic_local_target_poses",
        "masked_mimic_local_target_masks",
        "masked_mimic_local_target_times",
        "masked_mimic_local_target_poses_masks",
        "masked_mimic_style_poses",
        "masked_mimic_style_masks",
        "masked_mimic_style_velocities",
        "masked_mimic_style_frame_masks",
        "masked_mimic_velocity_command",
        "masked_mimic_velocity_command_mask",
    }
    assert keys == expected
    # The privileged encoder inputs and the internal latent are not host inputs.
    assert "max_coords_obs" not in keys
    assert "mimic_target_poses" not in keys
    assert VAE_LATENT_KEY not in keys
