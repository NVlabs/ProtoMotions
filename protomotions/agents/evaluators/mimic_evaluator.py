# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import torch
import numpy as np
from typing import Dict, Optional, Tuple, Any
from torch import Tensor
import math

from protomotions.agents.evaluators.base_evaluator import BaseEvaluator
from protomotions.agents.evaluators.metrics import MotionMetrics
from protomotions.components.motion_lib import MotionLib
from protomotions.agents.evaluators.config import MimicEvaluatorConfig
from protomotions.envs.motion_manager.mimic_motion_manager import MimicMotionManager


class MimicEvaluator(BaseEvaluator):
    """Evaluator for Mimic agent's motion tracking performance."""

    def __init__(self, agent: Any, fabric: Any, config: MimicEvaluatorConfig):
        """
        Initialize the Mimic evaluator.

        Args:
            agent: The Mimic agent to evaluate
            fabric: Lightning Fabric instance for distributed training
        """
        super().__init__(agent, fabric, config)

        # Base metric keys from config; do not mutate across evaluations
        self.base_eval_keys = config.eval_metric_keys.copy()

    @property
    def motion_lib(self) -> MotionLib:
        """Motion library (from agent)."""
        return self.agent.motion_lib

    @property
    def num_envs(self) -> int:
        """Number of environments (from agent)."""
        return self.agent.num_envs

    @property
    def motion_manager(self) -> MimicMotionManager:
        """Motion manager (from env)."""
        return self.env.motion_manager

    def _register_plugins(self) -> None:
        """Register metric computation plugins."""
        # Register smoothness evaluator as a plugin (now in base class)
        self._register_smoothness_plugin(window_sec=0.4, high_jerk_threshold=6500.0)

    def _create_metrics_dict(
        self,
        num_motions: int,
        motion_num_frames: Tensor,
        max_eval_steps: int,
        include_raw_robot_state: bool = True,
    ) -> Dict[str, MotionMetrics]:
        """Initialize the metrics dictionary for tracking evaluation progress.

        Creates MotionMetrics objects for base evaluation keys and optionally
        for raw robot state (for detailed analysis or saving).

        Args:
            num_motions: Total number of unique motions being evaluated.
            motion_num_frames: Tensor containing the frame count for each motion.
            max_eval_steps: Maximum number of steps to record per motion.
            include_raw_robot_state: If True, includes metrics for full robot
                state (positions, rotations, velocities).

        Returns:
            A dictionary mapping metric names to MotionMetrics objects.
        """
        # Use base class helper to create metrics from config keys
        metrics = self._create_base_metrics(
            self.base_eval_keys, num_motions, motion_num_frames, max_eval_steps
        )

        # Add raw robot state metrics if requested
        if include_raw_robot_state:
            self._add_robot_state_metrics(
                metrics, num_motions, motion_num_frames, max_eval_steps
            )

        return metrics

    def initialize_eval(self) -> Dict:
        """
        Initialize metrics dictionary with required keys.

        Returns:
            Dictionary of initialized MotionMetrics
        """
        num_motions = self.motion_lib.num_motions()
        motion_num_frames = self.motion_lib.get_motion_num_frames(None)
        motion_num_frames = motion_num_frames.clamp(max=self.config.max_eval_steps)

        return self._create_metrics_dict(
            num_motions,
            motion_num_frames,
            self.config.max_eval_steps,
            include_raw_robot_state=True,
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

    def _update_motion_sampling_weights(
        self,
        metrics: Dict[str, MotionMetrics],
        success_metric_key: str = "gt_err",
        failure_threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Update motion sampling weights based on success/failure rates.

        Args:
            metrics: Dictionary of evaluation metrics
            success_metric_key: Key to use for determining success/failure
            failure_threshold: Threshold above which motion is considered failed

        Returns:
            Dictionary with tracking_success_rate metric
        """
        assert success_metric_key in metrics, f"Metric '{success_metric_key}' not found"

        # Determine success/failure
        gt_err_max = metrics[success_metric_key].max_reduce_each_motion()
        tracking_failures = gt_err_max > failure_threshold
        tracking_success_rate = 1.0 - tracking_failures.float().mean().item()

        failed_motions = torch.nonzero(tracking_failures).flatten().tolist()
        success_motions = torch.nonzero(~tracking_failures).flatten().tolist()

        # Save failed motions to disk
        self._save_failed_motions(failed_motions, self.agent.current_epoch)

        # Update sampling weights
        success_discount = math.pow(0.999, self.config.eval_metrics_every)
        new_weights = self.env.motion_manager.motion_weights.clone()
        new_weights[success_motions] *= success_discount
        new_weights[failed_motions] /= success_discount
        new_weights.clamp_(min=0.03, max=1.0)
        self.env.motion_manager.update_sampling_weights(new_weights)

        return {"eval/tracking_success_rate": tracking_success_rate}

    def run_evaluation(self, metrics: Dict) -> None:
        """
        Run evaluation across multiple motions.

        Args:
            metrics: Dictionary to collect evaluation metrics
        """

        self._cached_robot_state = self.env.simulator.get_robot_state()
        self._cached_markers_state = self.env.get_markers_state()
        self._cached_env_actions = self.env.simulator.get_current_actions()
        self._cached_progress_buf = self.env.progress_buf.clone()
        self._cached_motion_ids = self.motion_manager.motion_ids.clone()
        self._cached_motion_times = self.motion_manager.motion_times.clone()
        self._cached_respawn_offset = self.env.respawn_root_offset.clone()

        # cache history buffers (cache all enabled observation history buffers)
        if self.env.self_obs_cb.humanoid_max_coords_obs_hist_buf is not None:
            self._cached_humanoid_max_coords_obs_hist = (
                self.env.self_obs_cb.humanoid_max_coords_obs_hist_buf.data.clone()
            )
        else:
            self._cached_humanoid_max_coords_obs_hist = None
        if self.env.self_obs_cb.humanoid_reduced_coords_obs_hist_buf is not None:
            self._cached_humanoid_reduced_coords_obs_hist = (
                self.env.self_obs_cb.humanoid_reduced_coords_obs_hist_buf.data.clone()
            )
        else:
            self._cached_humanoid_reduced_coords_obs_hist = None
        if self.env.self_obs_cb.previous_actions_hist_buf is not None:
            self._cached_previous_actions_hist = (
                self.env.self_obs_cb.previous_actions_hist_buf.data.clone()
            )
        else:
            self._cached_previous_actions_hist = None

        fixed_motion_ids, first_env_indices = (
            self.motion_manager.get_unique_fixed_motions()
        )

        if fixed_motion_ids.numel() > 0:
            # only evaluate fixed motions
            # NOTE: we do not support some envs having no fixed motions (no scene)
            # no such cases in current datasets, and also it
            # would be awkward to define what evaluator should do in that case
            print(
                f"Only evaluating fixed motions: {fixed_motion_ids}, First environment indices: {first_env_indices}"
            )
            self.evaluate_episode(metrics, first_env_indices, fixed_motion_ids)
            return

        num_motions = self.motion_lib.num_motions()

        for motion_id_start in range(0, num_motions, self.num_envs):
            motion_id_end = min(motion_id_start + self.num_envs, num_motions)
            motion_ids = torch.arange(
                motion_id_start, motion_id_end, device=self.device
            )
            env_ids = torch.arange(0, motion_ids.numel(), device=self.device)

            print(
                f"Evaluating motions {motion_id_start} to {motion_id_end}, out of total {num_motions} motions"
            )

            self.evaluate_episode(metrics, env_ids, motion_ids)

    def evaluate_episode(
        self,
        metrics: Dict,
        active_env_ids: Tensor,
        active_motion_ids: Tensor,
    ) -> None:
        """Evaluate a single episode for a batch of motions.

        Resets the environment with the specified motions and steps through
        the episode until completion or max steps, accumulating metrics.

        Args:
            metrics: Dictionary to collect evaluation metrics.
            active_env_ids: Tensor of environment IDs to use for this batch.
            active_motion_ids: Tensor of motion IDs to evaluate in these environments.
        """

        assert len(active_env_ids) == len(active_motion_ids)

        # Initialize environment for this episode
        self.motion_manager.motion_ids[active_env_ids] = active_motion_ids
        max_len = self.motion_lib.get_motion_num_frames(active_motion_ids).max().item()
        max_len = min(max_len, self.config.max_eval_steps)
        self.motion_manager.motion_times[active_env_ids] = 0.0

        # Reset the environment once at the beginning on flat terrain
        # disable_motion_resample=True to use the motion_ids we just set above
        obs, _ = self.env.reset(
            active_env_ids, sample_flat=True, disable_motion_resample=True
        )
        obs = self.agent.add_agent_info_to_obs(obs)
        obs_td = self.agent.obs_dict_to_tensordict(obs)

        # Run the episode and collect metrics (no resets during episode)
        for _ in range(max_len):
            # Obtain actor predictions
            model_outs = self.agent.model(obs_td)
            if "mean_action" in model_outs:
                actions = model_outs["mean_action"]
            else:
                actions = model_outs["action"]
            # Step the environment
            obs, rewards, dones, terminated, extras = self.env.step(actions)
            obs = self.agent.add_agent_info_to_obs(obs)

            obs_td = self.agent.obs_dict_to_tensordict(obs)
            # Update metrics
            self.update_metrics_from_env_extras(
                metrics, extras, active_env_ids, active_motion_ids
            )

    def add_extra_obs_to_agent(self, obs: Tensor):
        return obs

    def update_metrics_from_env_extras(
        self,
        metrics: Dict,
        extras: Dict,
        active_env_ids: Tensor,
        active_motion_ids: Tensor,
    ) -> None:
        """
        Update metrics from env.extras.

        Args:
            metrics: Dictionary to update with metrics
            extras: Dictionary of extra information from environment step
            motion_ids: Tensor of motion IDs being evaluated
        """

        assert len(active_env_ids) == len(active_motion_ids)

        # Use metrics dict as source of truth for which keys to update
        for k in metrics.keys():
            if f"mimic_other/{k}" in extras:
                value = extras[f"mimic_other/{k}"].detach()
            elif f"raw_r/{k}" in extras:
                value = extras[f"raw_r/{k}"].detach()
            # getting raw robot states in metrics for computation e.g. smoothness, etc.
            elif f"raw/{k}" in extras:
                value = extras[f"raw/{k}"].detach()
            else:
                raise ValueError(f"Key {k} not found in env.extras")

            metric = value[active_env_ids]  # in case there are more envs than motions

            metrics[k].update(active_motion_ids, metric)

    def process_eval_results(self, metrics: Dict) -> Tuple[Dict, Optional[float]]:
        """
        Process results and check for early termination.

        Args:
            metrics: Dictionary of collected metrics

        Returns:
            Tuple containing:
                - Dict of processed metrics for logging
                - Optional score value for determining best model
        """
        to_log = {}

        # Each rank uses its own tracking success rate to update the weights
        tracking_metrics = self._update_motion_sampling_weights(metrics)
        to_log.update(tracking_metrics)

        # Each rank computes its own scalar metrics from its local MotionMetrics
        # These scalars will be averaged across ranks by aggregate_scalar_metrics in post_epoch_logging
        # This avoids the memory overhead of merging full MotionMetrics objects across ranks

        # Log base metrics with mean/max/min aggregations
        base_metrics_log = self._gen_metrics(metrics, self.config.eval_metric_keys)
        to_log.update(base_metrics_log)

        # Compute additional metrics from plugins (e.g., smoothness)
        additional_metrics = self._compute_additional_metrics(metrics)
        to_log.update(additional_metrics)

        # Save the raw body pos and rot etc. at specified intervals
        if self.fabric.global_rank == 0:
            if (
                self.config.save_predicted_motion_lib_every is not None
                and self.eval_count % self.config.save_predicted_motion_lib_every == 0
            ):
                try:
                    self._save_predicted_motion_lib(
                        metrics, epoch=self.agent.current_epoch
                    )
                except Exception as e:
                    print(f"Warning: failed to save predicted MotionLib file: {e}")

        # these return can optionally be used to save the "best metric" model
        return to_log, to_log.get(
            "eval/tracking_success_rate", to_log.get("eval/gt_err", None)
        )

    def cleanup_after_evaluation(self) -> None:
        self.motion_manager.motion_ids = self._cached_motion_ids.clone()
        self.motion_manager.motion_times = self._cached_motion_times.clone()

        env_ids = torch.arange(0, self.num_envs, device=self.device)
        self.env.simulator.reset_envs(self._cached_robot_state, None, env_ids)

        # Restore history buffers (all enabled observation history buffers)
        if self.env.self_obs_cb.humanoid_max_coords_obs_hist_buf is not None:
            self.env.self_obs_cb.humanoid_max_coords_obs_hist_buf.data.copy_(
                self._cached_humanoid_max_coords_obs_hist
            )
        if self.env.self_obs_cb.humanoid_reduced_coords_obs_hist_buf is not None:
            self.env.self_obs_cb.humanoid_reduced_coords_obs_hist_buf.data.copy_(
                self._cached_humanoid_reduced_coords_obs_hist
            )
        if self.env.self_obs_cb.previous_actions_hist_buf is not None:
            self.env.self_obs_cb.previous_actions_hist_buf.data.copy_(
                self._cached_previous_actions_hist
            )

        self.env.progress_buf[env_ids] = self._cached_progress_buf[env_ids]
        self.env.reset_buf[env_ids] = False
        self.env.terminate_buf[env_ids] = False
        self.env.respawn_root_offset.copy_(self._cached_respawn_offset)

        # hack: if isaacgym
        if "isaacgym" in self.env.simulator.config._target_.lower():
            self.env.simulator.step(self._cached_env_actions, markers_callback=None)

        del self._cached_robot_state
        del self._cached_markers_state
        del self._cached_env_actions
        del self._cached_progress_buf
        del self._cached_motion_ids
        del self._cached_motion_times
        del self._cached_humanoid_max_coords_obs_hist
        del self._cached_humanoid_reduced_coords_obs_hist
        del self._cached_previous_actions_hist

    def simple_test_policy(self, collect_metrics: bool = False) -> None:
        """
        Evaluates the policy in evaluation mode.

        Args:
            collect_metrics: whether to collect metrics from the evaluation
            Will print the metrics to the console if collect_metrics is True
        """
        assert (
            self.fabric.world_size == 1
        ), "Simple test policy only supported for single process"

        num_motions = self.motion_lib.num_motions()
        motion_lengths = self.motion_lib.get_motion_length(None)
        motion_num_frames = (motion_lengths / self.env.dt).floor().long()

        if collect_metrics:
            metrics = self._create_metrics_dict(
                num_motions,
                motion_num_frames,
                motion_num_frames.max().item(),
                include_raw_robot_state=False,  # Simple test doesn't need robot state
            )
        else:
            metrics = None

        self.agent.eval()
        done_indices = None  # Force reset on first entry
        step = 0

        # Store actions for plotting (only for single motion evaluation)
        actions_storage = [] if collect_metrics and num_motions == 1 else None

        print("Evaluating policy...")
        try:
            while True:
                obs, _ = self.env.reset(done_indices)
                obs = self.agent.add_agent_info_to_obs(obs)
                obs_td = self.agent.obs_dict_to_tensordict(obs)

                cur_motion_ids = self.motion_manager.motion_ids
                cur_env_ids = torch.arange(
                    0, cur_motion_ids.numel(), device=self.device
                )

                # Obtain actor predictions
                model_outs = self.agent.model(obs_td)

                if "mean_action" in model_outs:
                    actions = model_outs["mean_action"]
                else:
                    actions = model_outs["action"]

                # Store actions for plotting (only for single motion and first environment)
                if (
                    actions_storage is not None
                    and len(actions_storage) < motion_num_frames.max().item()
                ):
                    actions_storage.append(
                        actions[0].detach().cpu().numpy()
                    )  # Store first env's actions

                # Step the environment
                obs, rewards, dones, terminated, extras = self.env.step(actions)
                obs = self.agent.add_agent_info_to_obs(obs)
                obs_td = self.agent.obs_dict_to_tensordict(obs)

                if collect_metrics:
                    # remove duplicate motions sampled
                    unique_motion_ids, first_indices = np.unique(
                        cur_motion_ids.cpu().numpy(), return_index=True
                    )
                    cur_env_ids = torch.from_numpy(first_indices).to(device=self.device)
                    cur_motion_ids = torch.from_numpy(unique_motion_ids).to(
                        device=self.device
                    )
                    self.update_metrics_from_env_extras(
                        metrics, extras, cur_env_ids, cur_motion_ids
                    )

                done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
                step += 1
        except KeyboardInterrupt:
            print("\nEvaluation interrupted by Ctrl+C, exiting...")
            if collect_metrics:
                print("Metrics up to now:")
                for k in (
                    self.config.eval_metric_keys
                ):  # do not reduce added eval keys for the raws
                    print(f"{k}: {metrics[k].mean_mean_reduce().item()}")

                # Plot per-frame metrics if only one motion
                if num_motions == 1:
                    self._plot_per_frame_metrics(metrics, actions_storage)
            return
        except Exception as e:
            print(f"Error in simple_test_policy: {e}")
            raise e

    def _plot_per_frame_metrics(
        self, metrics: Dict, actions_storage: list = None
    ) -> None:
        """
        Plot per-frame metrics vs time when evaluating a single motion.
        Uses base class plotting with custom colors for contact forces.

        Args:
            metrics: Dictionary of MotionMetrics objects
            actions_storage: List of action arrays for plotting (optional, currently unused)
        """
        # Define custom colors for specific metrics
        custom_colors = {}

        # Use base class generic plotting with custom colors
        super()._plot_per_frame_metrics(
            metrics,
            keys_to_plot=self.config.eval_metric_keys
            if hasattr(self.config, "eval_metric_keys")
            else None,
            custom_colors=custom_colors,
            output_filename="metrics_per_frame_plot.png",
        )

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
            "rigid_body_contacts",
        ]

        # Ensure required data exists
        for k in required_keys:
            if k not in metrics:
                raise ValueError(
                    f"Missing metric '{k}' required to build predicted MotionLib"
                )

        device = self.device
        num_motions = self.motion_lib.num_motions()

        # Use the per-motion valid frame counts from dof_pos (others should match)
        motion_num_frames = metrics["dof_pos"].motion_lens.to(device=device).long()
        assert (
            motion_num_frames.shape[0] == num_motions
        ), "motion_num_frames size mismatch"

        # Compute length_starts like MotionLib
        lengths_shifted = motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        length_starts = lengths_shifted.cumsum(0)

        motion_dt = (
            torch.ones(num_motions, dtype=torch.float32, device=device) * self.env.dt
        )
        motion_lengths = motion_num_frames.to(dtype=torch.float32) * self.env.dt

        # Helper to pack per-motion tensors into a single time-concatenated tensor
        def pack_metric(metric_key: str) -> torch.Tensor:
            data = metrics[metric_key].data  # [num_motions, max_frames, ...]
            per_motion = []
            for m in range(num_motions):
                f = motion_num_frames[m].item()
                # Clamp to available frames
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

        # Pack predicted contacts from metrics
        contacts_data = metrics[
            "rigid_body_contacts"
        ].data  # [num_motions, max_frames, num_bodies]
        contacts_list = []
        for m in range(num_motions):
            f = motion_num_frames[m].item()
            # Clamp to available frames
            f = min(f, contacts_data.shape[1])
            # Convert float contacts to bool for consistency with MotionLib format
            contacts_list.append(contacts_data[m, :f].bool().detach().clone())
        contacts = torch.cat(contacts_list, dim=0)

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
            "contacts": contacts,  # Always save predicted contacts
        }

        # create dir if not exists
        output_dir = self.root_dir / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"predicted_motion_lib_epoch_{epoch}.pt"
        torch.save(save_data, output_path)
        print(f"Predicted MotionLib saved to {output_path}")
