# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_mujoco_installation_uses_shipped_g1_example():
    installation = (
        REPO_ROOT / "docs/source/getting_started/installation.rst"
    ).read_text()
    checkpoint = Path("data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt")
    motion = Path("data/motion_for_trackers/g1_bones_seed_mini.pt")

    assert "results/experiment/last.ckpt" not in installation
    assert str(checkpoint) in installation
    assert str(motion) in installation
    assert "--simulator mujoco" in installation
    assert "--num-envs 1" in installation
    assert (REPO_ROOT / checkpoint).is_file()
    assert (REPO_ROOT / checkpoint.parent / "resolved_configs_inference.pt").is_file()
    assert (REPO_ROOT / motion).is_file()
