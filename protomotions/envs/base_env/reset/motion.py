# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which reference clip each env tracks: the motion library and its manager.

:class:`ReferenceMotion` is the one place a motion library is installed
(``install``): it validates the library against the robot, smooths its
contact labels when the env config asks for it, and builds the motion
manager over it. Everything else -- the spawn placer, the reset pose, and
whoever reaches the owner as ``env.reference_motion`` -- reads the CURRENT
library and manager through this object, because the live motion-shard
switcher (``agents/base_agent/motion_shards.py``) replaces both mid-run.

It is also the motion clock of the step walk (an ``Advancer``): ``advance``
ticks the current manager's ``post_physics_step`` when it has one (the mimic
manager does, the base manager does not). The env's advance list is static,
so a switch needs no rebuilding: the clock ticks whatever is installed.

On the snapshot walk (a ``SnapshotOwner``) it carries the clip assignment:
``save_state`` is the manager's ``(motion_ids, motion_times)`` as owned
clones (None without a manager) and ``load_state`` writes them back in
place, so the evaluator's excursion restores the clips training was playing
through the ordinary ``env.restore_state``.

``ref_reset_envs`` is the head of ``BaseEnv.reset``: which of the envs being
reset start from a reference clip, and the clips sampled for them. It is the
only caller of ``motion_manager.sample_motions`` on the reset path; the call
and its arguments are the ones the inline version made (the parity harness
logs them, so their order against the other random draws is fixed).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from torch import Tensor

from protomotions.utils.hydra_replacement import get_class


class ReferenceMotion:
    """The reference motion the envs track: the installed library, its manager
    (the step clock), and the reset-time choice of which envs start from a
    clip."""

    name = "motion"

    def __init__(
        self,
        manager_config,
        robot_config,
        scene_lib,
        num_envs: int,
        env_dt: float,
        device: torch.device,
        contact_smooth_window: int,
        generator: Optional[torch.Generator],
    ):
        self.config = manager_config
        self.robot_config = robot_config
        self.scene_lib = scene_lib
        self.num_envs = num_envs
        self.env_dt = env_dt
        self.device = device
        self.contact_smooth_window = contact_smooth_window
        # The motion-sampling stage's generator (None: the global stream).
        self.generator = generator
        self.motion_lib: Any = None
        self.motion_manager: Any = None

    def install(self, motion_lib) -> None:
        """Install a prebuilt motion library and build its manager.

        An incompatible library is refused before anything changes. An empty
        library (no motions) is installed without a manager.
        """
        if motion_lib.num_motions() > 0:
            self.validate_compatibility(motion_lib)

        if motion_lib.num_motions() > 0 and self.contact_smooth_window > 0:
            motion_lib.smooth_contacts(self.contact_smooth_window)

        self.motion_lib = motion_lib
        if motion_lib.num_motions() > 0:
            self.motion_manager = self.create_manager()
        else:
            self.motion_manager = None

    def clear(self) -> None:
        """Drop the installed library and its manager. The live shard switch
        (``agents/base_agent/motion_shards.py``) clears before building the
        replacement so nothing reads the old library in between; the next
        ``install`` fills both again."""
        self.motion_lib = None
        self.motion_manager = None

    def validate_compatibility(self, motion_lib) -> None:
        """Validate that the motion file is compatible with the robot config."""
        ki = self.robot_config.kinematic_info
        expected_dofs = ki.num_dofs
        expected_bodies = ki.num_bodies

        sample_state = motion_lib.get_motion_state(
            torch.zeros(1, dtype=torch.long, device=self.device),
            torch.zeros(1, device=self.device),
        )
        motion_dofs = sample_state.dof_pos.shape[1]
        motion_bodies = sample_state.rigid_body_pos.shape[1]

        if motion_dofs != expected_dofs or motion_bodies != expected_bodies:
            raise ValueError(
                f"\n{'=' * 70}\n"
                f"MOTION FILE / ROBOT MISMATCH\n"
                f"{'=' * 70}\n"
                f"Motion file has {motion_dofs} DOFs and {motion_bodies} bodies,\n"
                f"but robot '{type(self.robot_config).__name__}' expects "
                f"{expected_dofs} DOFs and {expected_bodies} bodies.\n\n"
                f"The motion file was likely generated for a different robot.\n"
                f"Make sure --motion-file matches the robot in your "
                f"checkpoint/config.\n"
                f"{'=' * 70}"
            )

    def create_manager(self):
        """Instantiate the motion manager of ``config`` over the installed library.

        With scenes that name a humanoid motion per scene, each env's clip is
        pinned to its scene's motion.
        """
        MotionManagerClass = get_class(self.config._target_)

        fixed_motion_ids = None
        if self.scene_lib.num_scenes() > 0:
            humanoid_motion_ids = self.scene_lib.get_humanoid_motion_ids()
            if humanoid_motion_ids is not None:
                fixed_motion_ids = torch.tensor(
                    humanoid_motion_ids, dtype=torch.long, device=self.device
                )

        return MotionManagerClass(
            config=self.config,
            num_envs=self.num_envs,
            env_dt=self.env_dt,
            device=self.device,
            motion_lib=self.motion_lib,
            fixed_motion_ids_per_env=fixed_motion_ids,
            generator=self.generator,
        )

    def advance(self, state) -> None:
        """Advance the reference motion's time by one env step (the clock);
        the robot state is not read here."""
        tick = getattr(self.motion_manager, "post_physics_step", None)
        if tick is not None:
            tick()

    def save_state(self) -> Optional[Tuple[Tensor, Tensor]]:
        """The clip assignment ``(motion_ids, motion_times)`` as owned clones;
        None without a manager."""
        if self.motion_manager is None:
            return None
        return self.motion_manager.get_assignment()

    def load_state(self, snapshot: Optional[Tuple[Tensor, Tensor]]) -> None:
        """Write a saved assignment back in place.

        None with no manager installed is the no-op it was saved as. A manager
        that appeared or disappeared since the save, or a clip id outside the
        installed library, means the library was switched between the save
        and the restore (the live shard switch): the snapshot belongs to
        another library and is refused rather than written through
        ``set_assignment``'s shape-only check.
        """
        if snapshot is None and self.motion_manager is None:
            return
        if snapshot is None or self.motion_manager is None:
            raise RuntimeError(
                "the clip assignment snapshot and the installed motion manager "
                "disagree: one has clips and the other has none (the motion "
                "library was switched between save_state and restore_state)"
            )
        motion_ids, motion_times = snapshot
        num_motions = self.motion_lib.num_motions()
        outside = (motion_ids < 0) | (motion_ids >= num_motions)
        if bool(outside.any()):
            raise ValueError(
                f"the clip assignment snapshot names clip "
                f"{int(motion_ids[outside][0])}, but the installed motion library "
                f"has {num_motions} clips (the motion library was switched between "
                "save_state and restore_state)"
            )
        self.motion_manager.set_assignment(motion_ids, motion_times)

    def ref_reset_envs(
        self,
        env_ids: Tensor,
        force_default_mask: Optional[Tensor],
        disable_motion_resample: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Which of ``env_ids`` reset from a reference clip, and their clips.

        Samples new clips for them (``motion_manager.sample_motions``) unless
        ``disable_motion_resample`` keeps the current assignment, which the
        evaluators use to replay specific motions. The control components
        are reset AFTER this, so they see the fresh ids and times.

        Returns ``(ref_env_ids, motion_ids, motion_times)``; the last two are
        None when no env resets from a clip.
        """
        # No motions - no ref resets
        if self.motion_lib.num_motions() == 0:
            empty_ids = torch.tensor([], device=self.device, dtype=torch.long)
            return empty_ids, None, None

        if force_default_mask is not None:
            assert (
                len(force_default_mask) == len(env_ids)
            ), f"force_default_mask length {len(force_default_mask)} != env_ids length {len(env_ids)}"
            ref_env_ids = env_ids[~force_default_mask]
        else:
            ref_env_ids = env_ids

        if len(ref_env_ids) > 0:
            if not disable_motion_resample:
                self.motion_manager.sample_motions(ref_env_ids)
            motion_ids = self.motion_manager.motion_ids[ref_env_ids]
            motion_times = self.motion_manager.motion_times[ref_env_ids]
        else:
            motion_ids = None
            motion_times = None

        return ref_env_ids, motion_ids, motion_times


__all__ = ["ReferenceMotion"]
