# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The reset chain's owners and helpers.

``BaseEnv.reset`` is a list of phases (``env.py`` declares ``RESET_PHASES``
and the reasons for their order); the objects here do the work of its head,
each one holding what it needs instead of the env (the recovery bank, the
reset pose and the spawn placer hold the simulator; none holds the env):

- :class:`~.motion.ReferenceMotion` -- the motion library and its manager
  (the only place a library is installed), the step clock, and the choice
  of which envs reset from a reference clip (``ref_reset_envs``).
- :class:`~.pose.ResetPose` -- the default, reference or mixed pose of the
  envs being reset, placed at their spawn.
- :class:`~.spawn.SpawnPlacer` -- the respawn offsets: sampled on the
  terrain or at the scene, applied to the pose, re-anchored on the robot
  after every step when the motion manager asks for it; snapshotted.
- :class:`~.placed_reference.PlacedReference` -- not a reset phase but the
  composition of the two above the tasks read: the reference clips' poses
  placed where each env spawned, and the clip assignment's writers
  (``env.reference``).
- :class:`~.recovery.RecoveryBank` -- the fall-pose cache generated at
  startup and the Bernoulli swap into it.
- :class:`~.perturbation.ResetPerturbation` -- the reset noise, under the
  env's perturbation switch.

Why the chain's order: clips are sampled before the pose is read from them;
the spawn is sampled with the pose about to land there (the height
correction needs the bodies); the recovery swap keeps the sampled XY and
needs the spawn's terrain height; the noise perturbs the final pose; one
simulator write follows, and everything after it (the row clears, the
read-back, the sensor reset, the history fill, the control components'
reset, the observations) sees the physical post-reset measurement. Every
random draw on the path (motion sampling, the terrain's spawn generator,
the recovery draw, the noise) keeps the position it always had.
"""

from protomotions.envs.base_env.reset.motion import ReferenceMotion
from protomotions.envs.base_env.reset.perturbation import ResetPerturbation
from protomotions.envs.base_env.reset.placed_reference import (
    PlacedReference,
    ReferenceWindow,
)
from protomotions.envs.base_env.reset.pose import ResetPose
from protomotions.envs.base_env.reset.recovery import RecoveryBank
from protomotions.envs.base_env.reset.spawn import SpawnPlacer

__all__ = [
    "PlacedReference",
    "RecoveryBank",
    "ReferenceMotion",
    "ReferenceWindow",
    "ResetPerturbation",
    "ResetPose",
    "SpawnPlacer",
]
