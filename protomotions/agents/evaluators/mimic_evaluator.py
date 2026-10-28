# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from protomotions.agents.evaluators.base_evaluator import BaseEvaluator
from protomotions.agents.evaluators.config import MimicEvaluatorConfig
from protomotions.agents.evaluators.metrics import MotionMetrics
from protomotions.components.motion_lib import MotionLib
from protomotions.envs.motion_manager.mimic_motion_manager import MimicMotionManager

logger = logging.getLogger(__name__)


@dataclass
class MimicEpisodeContext:
    """Per-episode-batch state for mimic evaluation."""

    motion_ids: Tensor  # which motion each env is tracking
    frame_limits: Tensor  # how many frames before clip ends


class MimicEvaluator(BaseEvaluator):
    """Evaluator for Mimic agent's motion tracking performance."""

    config: MimicEvaluatorConfig

    @property
    def motion_lib(self) -> MotionLib:
        """Motion library (from agent)."""
        return self.agent.motion_lib

    @property
    def motion_manager(self) -> MimicMotionManager:
        """Motion manager (from env)."""
        return self.env.reference_motion.motion_manager

    def _register_plugins(self) -> None:
        """Register metric computation plugins."""
        self._register_smoothness_plugin(window_sec=0.4, high_jerk_threshold=6500.0)
        self._register_action_smoothness_plugin()

    def _create_metrics(
        self,
        num_motions: int,
        motion_num_frames: Tensor,
        max_eval_steps: int,
    ) -> Dict[str, MotionMetrics]:
        """Create MotionMetrics buffers for trajectory collection (robot state + actions)."""
        metrics = {}

        self._add_robot_state_metrics(
            metrics, num_motions, motion_num_frames, max_eval_steps
        )

        num_dofs = self.env.robot_config.kinematic_info.num_dofs
        metrics["actions"] = MotionMetrics(
            num_motions, motion_num_frames, max_eval_steps, num_dofs, device=self.device
        )

        return metrics

    def initialize_eval(self) -> Dict:
        """Initialize evaluation tracking and cache the state eval will overwrite.

        Saves the env state (the clip assignment is part of it) and the model's
        rollout context, both restored by ``cleanup_after_evaluation``.
        """
        num_motions = self.motion_lib.num_motions()
        motion_lengths = self.motion_lib.get_motion_length(None)
        motion_num_frames = (motion_lengths / self.env.dt).floor().long()
        motion_num_frames = motion_num_frames.clamp(max=self.config.max_eval_steps)
        self._init_eval_component_buffers(num_motions)

        # Cache env and model rollout context (restored in
        # cleanup_after_evaluation). Evaluation reseeds the model context of every
        # env it resets, so training would otherwise resume with eval's context.
        self._save_training_state()

        return self._create_metrics(
            num_motions, motion_num_frames, self.config.max_eval_steps
        )

    def _save_failed_motions(self, failed_motions: list, epoch: int) -> None:
        """
        Save list of failed motions to a text file.

        Args:
            failed_motions: List of motion IDs that failed tracking
            epoch: Current epoch number
        """
        filename = f"failed_motions_epoch_{epoch}_rank_{self.fabric.global_rank}.txt"
        self._save_list_to_file(failed_motions, filename, subdirectory="failed_motions")

    def _update_motion_sampling_weights(self, result) -> None:
        """Update motion sampling weights based on evaluation component failures."""
        if result.evaluated is None or not self.config.failure_components:
            return

        failed_motions = torch.nonzero(result.failed & result.evaluated).flatten().tolist()
        success_motions = torch.nonzero(~result.failed & result.evaluated).flatten().tolist()
        if not failed_motions and not success_motions:
            return

        self._save_failed_motions(failed_motions, self.agent.current_epoch)

        success_discount = math.pow(
            self.config.motion_weights_rules.motion_weights_update_success_discount,
            self.config.eval_metrics_every,
        )
        failure_discount = math.pow(
            self.config.motion_weights_rules.motion_weights_update_failure_discount,
            self.config.eval_metrics_every,
        )
        new_weights = self.env.reference_motion.motion_manager.motion_weights.clone()
        new_weights[success_motions] *= success_discount
        if failure_discount != 0:
            new_weights[failed_motions] /= failure_discount
        else:
            new_weights[failed_motions] = 1.0
        # The writer floors every eligible clip and keeps the ineligible at 0.
        self.env.reference_motion.motion_manager.set_sampling_weights(
            new_weights,
            floor=self._resolved_min_motion_weight(new_weights.shape[0]),
            ceiling=float("inf"),
        )

    def _resolved_min_motion_weight(self, num_motions: int) -> float:
        """Resolve ``min_motion_weight``, which may be the string '1/num_motions'.

        Without this floor the success discount compounds without bound: at the
        default 0.999 with ``eval_metrics_every=200`` each update multiplies a
        succeeding motion's weight by 0.82, so after ~80 updates it sits at ~1e-7
        while any motion that failed once is reset to 1.0. Sampling then collapses
        onto whichever handful of clips failed most recently, and every
        distribution-sensitive metric starts tracking that churn rather than the
        policy. The floor keeps the curriculum a re-weighting rather than a
        replacement of the training set.
        """
        min_weight = self.config.motion_weights_rules.min_motion_weight
        if isinstance(min_weight, str):
            if min_weight == "1/num_motions":
                return 1.0 / max(num_motions, 1)
            return float(min_weight)
        return float(min_weight)

    def _reset_episode(self, env_ids: torch.Tensor):
        """Reset the full physics batch while scoring only unique motion IDs.

        Evaluation is an ordinary rollout in which only ``env_ids`` are scored. Every
        other env plays its background clip (see ``_background_motion_ids``) and, like
        a training env, restarts it whenever it falls or the clip ends. No env is
        parked or left idle: all envs share one physics scene, so the unscored ones
        should load it the way training does. Parking them below the floor instead
        cost ~12.8 points of SAMP eval success at 8192 envs (75.96% vs 88.80%).
        Scoring stays on the deduped ``env_ids`` subset -- widening it would corrupt
        the motion-id-indexed buffers (duplicate indices drop writes).

        Every env reset here, scored or not, also gets fresh model rollout context
        (for example MaskedMimic's per-episode latent noise), as training does after
        an env finishes. ``env.reset`` already clears the env's own history.
        """
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        return super()._reset_episode(all_env_ids)

    def _reset_envs(self, env_ids: Tensor):
        self.motion_manager.assign(
            env_ids,
            motion_times=torch.zeros_like(
                env_ids, dtype=self.motion_manager.motion_times.dtype
            ),
        )
        return super()._reset_envs(env_ids)

    def run_evaluation(self) -> None:
        """Run evaluation across multiple motions."""
        for env_ids, motion_ids in self._build_eval_batches():
            motion_lengths = self.motion_lib.get_motion_length(motion_ids)
            max_len = min(
                (motion_lengths.max() / self.env.dt).floor().long().item(),
                self.config.max_eval_steps,
            )
            # Build episode context before evaluate_episode so hooks can read it
            self._episode_ctx = MimicEpisodeContext(
                motion_ids=motion_ids,
                frame_limits=(motion_lengths / self.env.dt)
                .floor()
                .long()
                .clamp(max=self.config.max_eval_steps),
            )
            self.evaluate_episode(env_ids, max_len)

    def _build_eval_batches(self):
        """Build list of (env_ids, motion_ids) batches to evaluate.

        Returns:
            List of (env_ids, motion_ids) tuples
        """
        fixed_motion_ids, first_env_indices = (
            self.motion_manager.get_unique_fixed_motions()
        )

        if fixed_motion_ids.numel() > 0:
            print(f"Only evaluating fixed motions: {fixed_motion_ids}")
            return [(first_env_indices, fixed_motion_ids)]

        num_motions = self.motion_lib.num_motions()
        batches = []
        for start in range(0, num_motions, self.num_envs):
            end = min(start + self.num_envs, num_motions)
            motion_ids = torch.arange(start, end, device=self.device)
            env_ids = torch.arange(0, motion_ids.numel(), device=self.device)
            print(f"Evaluating motions {start} to {end}, out of total {num_motions}")
            batches.append((env_ids, motion_ids))
        return batches

    # --- Hook overrides ---

    def _background_motion_ids(self) -> Tensor:
        """The clip each env plays while it is not being scored.

        An env with a fixed (scene-paired) clip plays that clip, the only one its
        scene objects match. Any other env plays library clip ``env_id % num_motions``:
        a fixed choice rather than a random draw, so evaluation stays reproducible,
        and one that spreads the unscored envs over different clips.
        """
        env_ids = torch.arange(self.num_envs, device=self.device)
        motion_ids = env_ids % self.motion_lib.num_motions()
        fixed = self.motion_manager.fixed_motion_ids_per_env
        if fixed is not None:
            motion_ids = torch.where(fixed != -1, fixed, motion_ids)
        return motion_ids

    def _on_episode_start(self, env_ids: Tensor) -> None:
        """Set motion_ids/times in the motion manager before reset.

        Every env starts its background clip from frame 0, then the scored
        subset is overlaid with this batch's motions.
        """
        self.motion_manager.set_assignment(
            self._background_motion_ids(),
            torch.zeros_like(self.motion_manager.motion_times),
        )
        self.motion_manager.assign(env_ids, motion_ids=self._episode_ctx.motion_ids)

    def _get_reset_kwargs(self) -> dict:
        """Customize env.reset() for mimic evaluation."""
        return {"sample_flat": True, "disable_motion_resample": True}

    def _scored_trials(self, env_ids: Tensor, step_idx: int):
        """Score each motion only until its window ends or it fails."""
        motion_ids = self._episode_ctx.motion_ids
        active = (self._episode_ctx.frame_limits > step_idx) & ~self._evaluation.failed[motion_ids]
        return env_ids[active], motion_ids[active]

    def _on_episode_step(self, env_ids: Tensor, trial_ids: Tensor, extras: Dict, actions: Tensor) -> None:
        """Collect smoothness metrics each step."""
        self._record_trajectory_step(
            self._metrics, extras, env_ids, trial_ids, actions
        )

    def _record_trajectory_step(
        self,
        metrics: Dict,
        extras: Dict,
        active_env_ids: Tensor,
        active_motion_ids: Tensor,
        actions: Tensor,
    ) -> None:
        """Record robot state and actions into trajectory buffers for this step."""
        if "actions" in metrics and actions is not None:
            metrics["actions"].update(
                active_motion_ids, actions[active_env_ids].detach()
            )

        for k in metrics.keys():
            if k == "actions":
                continue
            if f"raw/{k}" in extras:
                metrics[k].update(
                    active_motion_ids, extras[f"raw/{k}"][active_env_ids].detach()
                )

    def process_eval_results(self) -> Tuple[Dict, Optional[float], int]:
        """Process results without changing training's sampling distribution."""
        to_log, success_rate, num_eval_items = super().process_eval_results()

        additional_metrics = self._compute_additional_metrics(self._metrics)
        to_log.update(additional_metrics)

        if self.fabric.global_rank == 0:
            if (
                self.config.save_predicted_motion_lib_every is not None
                and self.eval_count % self.config.save_predicted_motion_lib_every == 0
            ):
                self._save_predicted_motion_lib(
                    self._metrics, epoch=self.agent.current_epoch
                )

        return to_log, success_rate, num_eval_items

    def update_training_state(self, result):
        self._update_motion_sampling_weights(result)

    def _save_predicted_motion_lib(
        self, metrics: Dict[str, MotionMetrics], epoch: int
    ) -> None:
        """Pack collected predicted metrics and save as a MotionLib-compatible .pt file.

        This creates a "predicted" version of MotionLib where unknown fields are copied
        from the ground-truth self.motion_lib.

        Args:
            metrics: Dictionary of MotionMetrics objects containing predicted data
            epoch: Current epoch number for filename
        """
        required_keys = [
            "dof_pos",
            "dof_vel",
            "rigid_body_pos",
            "rigid_body_rot",
            "rigid_body_vel",
            "rigid_body_ang_vel",
        ]
        if self.env.simulator.has_complete_contact_labels:
            required_keys.append("rigid_body_contacts")

        # Ensure required data exists
        for k in required_keys:
            if k not in metrics:
                raise ValueError(
                    f"Missing metric '{k}' required to build predicted MotionLib"
                )

        device = self.device
        num_motions = self.motion_lib.num_motions()

        motion_num_frames = metrics["dof_pos"].motion_lens.to(device=device).long()
        assert (
            motion_num_frames.shape[0] == num_motions
        ), "motion_num_frames size mismatch"

        # Mask motions that were never rolled out, so the saved lib doesn't
        # contain zero-filled phantom frames that trip the playback assertion.
        # frame_counts is incremented by MotionMetrics.update; zero here means
        # no env ever wrote a frame for this motion.  Motion IDs stay aligned
        # with the GT lib; un-rolled motions simply have length 0 in the saved file.
        rolled_out = metrics["dof_pos"].frame_counts.to(device=device) > 0
        num_skipped = int((~rolled_out).sum().item())
        if num_skipped > 0:
            print(
                f"Predicted MotionLib: masking {num_skipped} / {num_motions} "
                f"un-rolled-out motions (length set to 0)"
            )
        # Every packed channel must contain that frame; a scheduled length is
        # only an upper bound. Never export an unwritten zero-filled tail.
        for key in required_keys:
            metric = metrics[key]
            motion_num_frames = torch.minimum(
                motion_num_frames,
                metric.frame_counts.to(device=device).long().clamp(max=metric.data.shape[1]),
            )

        lengths_shifted = motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        length_starts = lengths_shifted.cumsum(0)

        motion_dt = (
            torch.ones(num_motions, dtype=torch.float32, device=device) * self.env.dt
        )
        motion_lengths = motion_num_frames.to(dtype=torch.float32) * self.env.dt

        def pack_metric(metric_key: str) -> torch.Tensor:
            data = metrics[metric_key].data
            per_motion = []
            for m in range(num_motions):
                f = motion_num_frames[m].item()
                f = min(f, data.shape[1])
                per_motion.append(data[m, :f].detach().clone())
            return torch.cat(per_motion, dim=0)

        # Build packed tensors matching MotionLib field names
        dps = pack_metric("dof_pos")  # [total_frames, num_dofs]
        dvs = pack_metric("dof_vel")  # [total_frames, num_dofs]

        # Rigid body tensors are stored flattened in metrics; reshape to [*, num_bodies, C]
        num_bodies = self.env.robot_config.kinematic_info.num_bodies
        gts_flat = pack_metric("rigid_body_pos")  # [total_frames, num_bodies*3]
        grs_flat = pack_metric("rigid_body_rot")  # [total_frames, num_bodies*4]
        gvs_flat = pack_metric("rigid_body_vel")  # [total_frames, num_bodies*3]
        gavs_flat = pack_metric("rigid_body_ang_vel")  # [total_frames, num_bodies*3]

        # Validate and reshape
        assert (
            gts_flat.shape[-1] == num_bodies * 3
        ), f"rigid_body_pos dim mismatch: {gts_flat.shape[-1]} vs {num_bodies*3}"
        assert (
            grs_flat.shape[-1] == num_bodies * 4
        ), f"rigid_body_rot dim mismatch: {grs_flat.shape[-1]} vs {num_bodies*4}"
        assert (
            gvs_flat.shape[-1] == num_bodies * 3
        ), f"rigid_body_vel dim mismatch: {gvs_flat.shape[-1]} vs {num_bodies*3}"
        assert (
            gavs_flat.shape[-1] == num_bodies * 3
        ), f"rigid_body_ang_vel dim mismatch: {gavs_flat.shape[-1]} vs {num_bodies*3}"

        gts = gts_flat.view(-1, num_bodies, 3)
        grs = grs_flat.view(-1, num_bodies, 4)
        gvs = gvs_flat.view(-1, num_bodies, 3)
        gavs = gavs_flat.view(-1, num_bodies, 3)

        # The recorded positions are in the simulator's world frame, so they
        # carry each env's spawn placement. A replay adds back the scene XY
        # and the terrain height but not the spawn-only height bump, so exactly
        # that is removed here (``PlacedReference.spawn_translation``): the saved clip
        # keeps the "drop from the bump, then settle" trajectory the policy
        # experienced. Velocities and rotations are translation-invariant.
        per_motion_offset = torch.zeros(num_motions, 3, device=device, dtype=gts.dtype)
        unique_motion_ids, first_env_indices = (
            self.motion_manager.get_unique_fixed_motions()
        )
        if unique_motion_ids.numel() > 0:
            per_motion_offset[unique_motion_ids] = self.env.reference.spawn_translation(
                first_env_indices
            ).to(device=device, dtype=gts.dtype)
        for m in range(num_motions):
            nframes = int(motion_num_frames[m].item())
            if nframes == 0:
                continue
            start = int(length_starts[m].item())
            gts[start : start + nframes] -= per_motion_offset[m].view(1, 1, 3)

        # Copy ground-truth motion weights and files
        gt_lib = self.motion_lib
        motion_weights = getattr(
            gt_lib,
            "motion_weights",
            torch.ones(num_motions, dtype=torch.float32, device=device),
        )
        motion_files = getattr(
            gt_lib,
            "motion_files",
            tuple([f"predicted_motion_{i}" for i in range(num_motions)]),
        )

        save_data = {
            "gts": gts,
            "grs": grs,
            "gvs": gvs,
            "gavs": gavs,
            "dvs": dvs,
            "dps": dps,
            "length_starts": length_starts,
            "motion_lengths": motion_lengths,
            "motion_dt": motion_dt,
            "motion_num_frames": motion_num_frames,
            "motion_weights": motion_weights,
            "motion_files": motion_files,
            "contacts": None,
        }
        # Disabled contact reporting means unknown labels, not all-false labels.
        if self.env.simulator.has_complete_contact_labels:
            save_data["contacts"] = pack_metric("rigid_body_contacts").bool()

        # create dir if not exists
        output_dir = self.root_dir / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"predicted_motion_lib_epoch_{epoch}.pt"
        torch.save(save_data, output_path)
        print(f"Predicted MotionLib saved to {output_path}")
