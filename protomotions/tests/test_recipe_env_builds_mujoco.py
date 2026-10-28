# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One real environment build of a shipped recipe: the configs training builds,
the components, ``BaseEnv``, a reset and a step, on the MuJoCo backend with one
environment.

The config-only check, the recipe-config tests and the parity scenarios each
stop short of something a training run does: ``mimic/mlp.py`` names its foot
contacts with a ``BodySelection`` in a reward, and the build's check once
refused that as unresolved only in a real build (the parity harness drops that
reward for its SMPL motions). This test runs the construction path end to end,
as ``train_agent.py`` does, in the ordinary suite; ``test_recipe_stages.py``
builds every recipe the same way behind its slow flag. Skipped without the LFS
payload of the motion file.
"""

import pytest
import torch

mujoco = pytest.importorskip("mujoco")

import protomotions.envs.component_manager as component_manager  # noqa: E402
from protomotions.tests.support.recipe_stages import (  # noqa: E402
    MOTION_FILE,
    build_reset_step,
    has_payload,
)


@pytest.mark.skipif(
    not has_payload(MOTION_FILE),
    reason="needs the LFS payload of data/motion_for_trackers/g1_bones_seed_mini.pt",
)
def test_mimic_mlp_builds_resets_and_steps_on_mujoco(monkeypatch):
    """The public tracker recipe constructs, and its named foot contacts reach
    the reward manager as an index tensor while the config keeps the names."""
    # Eager kernels: compiling the components dominates the build time on CPU.
    monkeypatch.setattr(component_manager, "TORCH_COMPILE_AVAILABLE", False)
    env, obs, rewards = build_reset_step("examples/experiments/mimic/mlp.py")

    contact_bodies = env._rewards.components["contact_match_rew"].static_params[
        "contact_body_ids"
    ]
    assert contact_bodies.dtype == torch.long and contact_bodies.numel() == 2
    assert rewards.shape == (1,) and torch.isfinite(rewards).all()
    assert all(torch.isfinite(value).all() for value in obs.values())
