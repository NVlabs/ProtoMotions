# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared observation context used by GPC prior and PEFT task experiments."""
from __future__ import annotations

from protomotions.robot_configs.base import RobotConfig


NEAREST_SURFACE_TERRAIN_HORIZONTAL_SCALE = 0.1


def nearest_surface_body_ids(robot_cfg: RobotConfig) -> list[int]:
    return list(robot_cfg.body_indices(*robot_cfg.trackable_bodies_subset))


def nearest_surface_obs_params(robot_cfg: RobotConfig, *, with_objects: bool) -> dict:
    """Keyword arguments binding ``compute_nearest_surface_vectors`` for a recipe.

    The kernel binds the scene-object surfaces (``scene.*``) by default; a run
    without a scene file passes ``with_objects=False`` and the four object
    inputs are unbound (``None``), so only the terrain is a surface and no
    scene name is bound.
    """
    params = {
        "terrain_horizontal_scale": NEAREST_SURFACE_TERRAIN_HORIZONTAL_SCALE,
        "body_ids": nearest_surface_body_ids(robot_cfg),
    }
    if not with_objects:
        params.update(
            object_pos=None,
            object_rot=None,
            neutral_pointclouds=None,
            object_valid_mask=None,
        )
    return params


PEFT_SAMPLING_MODE_PRIOR = "prior_constraint"
PEFT_SAMPLING_MODE_NUCLEUS = "nucleus"
PEFT_SAMPLING_MODE_CHOICES = (
    PEFT_SAMPLING_MODE_PRIOR,
    PEFT_SAMPLING_MODE_NUCLEUS,
)


def add_peft_sampling_mode_argument(parser):
    parser.add_argument(
        "--peft-sampling-mode",
        choices=PEFT_SAMPLING_MODE_CHOICES,
        default=PEFT_SAMPLING_MODE_PRIOR,
        help=(
            "How PEFT rollouts sample tokens: 'prior_constraint' limits them to "
            "the reference policy's top-p set, while 'nucleus' samples from the "
            "student nucleus and regularizes toward the reference with KL. The "
            "reference is the policy a new run starts from, whatever checkpoint "
            "that is, such as the loaded SFT policy on a warm start; a resume "
            "keeps it. --overrides "
            "agent.model.actor.peft.reference_checkpoint=<file> pins the "
            "trained policy in another checkpoint instead."
        ),
    )


def peft_sampling_mode_kwargs(args):
    sampling_mode = getattr(
        args,
        "peft_sampling_mode",
        PEFT_SAMPLING_MODE_PRIOR,
    )
    if sampling_mode == PEFT_SAMPLING_MODE_PRIOR:
        return {
            "sampling_mode": "prior_constraint",
            "top_p": 1.0,
            "prior_top_p": 0.9,
            "kl_coeff": 0.0,
        }
    if sampling_mode == PEFT_SAMPLING_MODE_NUCLEUS:
        return {
            "sampling_mode": "nucleus",
            "top_p": 0.9,
            "prior_top_p": 1.0,
            "kl_coeff": 0.1,
        }
    raise ValueError(f"Unknown PEFT sampling mode: {sampling_mode}")
