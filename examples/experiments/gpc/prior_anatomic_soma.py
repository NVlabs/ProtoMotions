# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train a GPC prior over the anatomical SOMA tracker's FSQ codes.

The prior itself is ``gpc/prior.py``: a transformer over the frozen tracker's
FSQ tokens, conditioned on ``max_coords_obs``, with the tracker's decoder
embedded. What changes is the world the tracker is rolled out in, which must
be the one it was trained in (``mimic/anatomical_soma_fsq_mimic.py``):
BeyondMimic quarter-effort actions instead of SOMA's tanh PD mapping, self
collisions, friction 1.0 and the G1-style physical randomization, with live
shard cycling for a ``slurmrank`` corpus. Motions must be retargeted to
``anatomic_soma``; the tracker checkpoint is the shipped
``motion_tracker/anatomic_soma_bones_fsq``.
"""

from examples.experiments.gpc import prior
from examples.experiments.gpc.prior import (
    additional_experiment_arguments as additional_experiment_arguments,
    agent_config as agent_config,
    scene_lib_config as scene_lib_config,
)
from examples.experiments.mimic import anatomical_soma_fsq_mimic as tracker
from examples.experiments.mimic.anatomical_soma_fsq_mimic import (
    apply_inference_overrides as apply_inference_overrides,
    configure_robot_and_simulator as configure_robot_and_simulator,
    motion_lib_config as motion_lib_config,
    terrain_config as terrain_config,
)
from protomotions.envs.action import make_bm_pd_action_config


def env_config(robot_cfg, args):
    cfg = prior.env_config(robot_cfg, args)
    cfg.action_config = make_bm_pd_action_config(
        robot_cfg, effort_fraction=tracker.EFFORT_FRACTION
    )
    return cfg
