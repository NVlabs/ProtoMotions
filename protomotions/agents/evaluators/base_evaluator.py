# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
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
"""Base evaluator for agent evaluation and metrics computation.

This module provides the base evaluation infrastructure for computing performance
metrics during training and evaluation. Evaluators run periodic assessments of
agent performance and compute task-specific metrics.

Key Classes:
    - BaseEvaluator: Base class for all evaluators with hook-based customization

Key Features:
    - Periodic evaluation during training
    - Hook pattern for subclass customization (4 hooks: start, reset_kwargs, check, step)
    - MdpComponent-based evaluation with threshold failure detection
    - Aggregate metrics via plugin system (see aggregate_metrics.py)
    - Episode statistics aggregation
    - Distributed evaluation support
    
Note:
    Aggregate metric plugins (SmoothnessAggregateMetric, ActionSmoothnessAggregateMetric)
    are defined in aggregate_metrics.py and compute post-hoc statistics over
    accumulated MotionMetrics trajectories.
"""

import logging
import numpy as np
import torch
from torch import Tensor
from typing import Dict, Optional, Tuple, Any
from lightning.fabric import Fabric

from protomotions.agents.evaluators.metrics import MotionMetrics
from protomotions.envs.base_env.env import BaseEnv
from protomotions.envs.component_manager import ComponentManager
from protomotions.envs.base_env.utils import combine_evaluation
from protomotions.agents.evaluators.config import EvaluatorConfig
from protomotions.agents.evaluators.aggregate_metrics import (
    SmoothnessAggregateMetric,
    ActionSmoothnessAggregateMetric,
)

log = logging.getLogger(__name__)


class BaseEvaluator:
    """Base class for agent evaluation and metrics computation.

    Runs periodic evaluations during training to assess agent performance.
    Collects episode statistics, computes task-specific metrics, and provides
    feedback for checkpoint selection (best model saving).

    Args:
        agent: The agent being evaluated.
        fabric: Lightning Fabric instance for distributed evaluation.
        config: Evaluator configuration specifying eval frequency and length.

    Example:
        >>> evaluator = BaseEvaluator(agent, fabric, config)
        >>> metrics, score = evaluator.evaluate()
    """

    def __init__(self, agent: Any, fabric: Fabric, config: EvaluatorConfig):
        """
        Initialize the evaluator.

        Args:
            agent: The agent to evaluate
            fabric: Lightning Fabric instance for distributed training
        """
        self.agent = agent
        self.fabric = fabric
        self.config = config

        self.metric_plugins = []
        self._register_plugins()
        self.eval_count = 0

        self._component_manager: Optional[ComponentManager] = None
        self._motion_failed: Optional[Tensor] = None
        self._per_component_failures: Dict[str, Tensor] = {}
        self._component_value_sum: Dict[str, Tensor] = {}
        self._component_value_min: Dict[str, Tensor] = {}
        self._component_value_max: Dict[str, Tensor] = {}
        self._component_step_count: Dict[str, Tensor] = {}
        
        # Instance state for metrics collection during evaluation
        self._metrics: Optional[Dict] = None

    @property
    def device(self) -> torch.device:
        """Device for computations (from fabric)."""
        return self.fabric.device

    @property
    def env(self) -> BaseEnv:
        """Environment instance (from agent)."""
        return self.agent.env

    @property
    def root_dir(self):
        """Root directory for saving outputs (from agent)."""
        return self.agent.root_dir

    @torch.no_grad()
    def evaluate(self) -> Tuple[Dict, Optional[float]]:
        """
        Evaluate the agent and calculate metrics.
        This is the main entry point that orchestrates the evaluation process.

        Returns:
            Tuple containing:
                - Dict of evaluation metrics for logging
                - Optional score value for determining best model
        """
        if not self.config.evaluation_components:
            return {}, None

        self.agent.eval()
        self._metrics = self.initialize_eval()
        if self._metrics is None:
            return {}, None

        self.run_evaluation()
        evaluation_log, evaluated_score = self.process_eval_results()
        self.cleanup_after_evaluation()
        self.eval_count += 1

        return evaluation_log, evaluated_score

    @property
    def num_envs(self) -> int:
        """Number of environments (from agent)."""
        return self.agent.num_envs

    @property
    def max_eval_steps(self) -> int:
        """Maximum steps per evaluation episode."""
        return self.config.max_eval_steps

    def initialize_eval(self) -> Dict:
        """Initialize evaluation tracking."""
        self._init_eval_component_buffers(self.num_envs)
        return {}

    def run_evaluation(self) -> None:
        """Run the evaluation process."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.evaluate_episode(env_ids, self.max_eval_steps)

    def evaluate_episode(self, env_ids: Tensor, max_steps: int) -> None:
        """Run a single episode batch.
        
        Subclasses customize behavior via 4 hooks:
        - _on_episode_start: pre-reset setup
        - _get_reset_kwargs: customize env.reset() call
        - _check_eval_components: per-step evaluation component checking
        - _on_episode_step: per-step data collection
        
        Args:
            env_ids: Environment IDs to evaluate [num_envs]
            max_steps: Maximum steps for this episode
        """
        self._on_episode_start(env_ids)
        
        obs, _ = self.env.reset(env_ids, **self._get_reset_kwargs())
        obs = self.agent.add_agent_info_to_obs(obs)
        obs_td = self.agent.obs_dict_to_tensordict(obs)
        
        for step_idx in range(max_steps):
            model_outs = self.agent.model(obs_td)
            actions = model_outs.get("mean_action", model_outs.get("action"))
            
            obs, rewards, dones, terminated, extras = self.env.step(actions)
            obs = self.agent.add_agent_info_to_obs(obs)
            obs_td = self.agent.obs_dict_to_tensordict(obs)
            
            self._check_eval_components(env_ids, step_idx)
            self._on_episode_step(env_ids, extras, actions)

    def _on_episode_start(self, env_ids: Tensor) -> None:
        """Hook called before episode reset. Override in subclasses for pre-reset setup.
        
        Args:
            env_ids: Environment IDs about to be reset
        """
        pass

    def _get_reset_kwargs(self) -> dict:
        """Hook to provide extra kwargs for env.reset(). Override in subclasses.
        
        Returns:
            Dictionary of kwargs passed to env.reset()
        """
        return {}

    def _check_eval_components(self, env_ids: Tensor, step_idx: int) -> None:
        """Hook for per-step evaluation component checking. Override in subclasses.
        
        Default behavior: check all env_ids, mapping env_ids to eval_ids 1:1.
        Subclasses can filter env_ids (e.g., skip finished motion clips) and
        provide custom env-to-eval-ID mapping.
        
        Args:
            env_ids: Environment IDs active this step
            step_idx: Current step index in the episode
        """
        self._check_evaluation_failures(env_ids, env_ids)

    def _on_episode_step(self, env_ids: Tensor, extras: Dict, actions: Tensor) -> None:
        """Hook called after each step. Override in subclasses to collect metrics.
        
        Args:
            env_ids: Environment IDs active this step
            extras: Extra data from env.step()
            actions: Actions taken this step
        """
        pass

    def process_eval_results(self) -> Tuple[Dict, Optional[float]]:
        """Process collected metrics and prepare for logging."""
        to_log = {}

        if self._motion_failed is not None:
            success_rate = 1.0 - self._motion_failed.float().mean().item()
            to_log["eval/success_rate"] = success_rate

            for name, component in self.config.evaluation_components.items():
                threshold = component.static_params.get("threshold", None)
                if threshold is not None:
                    failure_rate = self._per_component_failures[name].float().mean().item()
                    to_log[f"eval/{name}/failure_rate"] = failure_rate

            for name in self._component_value_sum.keys():
                step_count = self._component_step_count[name].float()
                valid = step_count > 0

                if valid.any():
                    mean_per_motion = self._component_value_sum[name] / step_count.clamp(min=1)
                    to_log[f"eval/{name}/mean"] = mean_per_motion[valid].mean().item()
                    to_log[f"eval/{name}/max"] = self._component_value_max[name][valid].max().item()
                    to_log[f"eval/{name}/min"] = self._component_value_min[name][valid].min().item()

            return to_log, success_rate

        return to_log, None

    def cleanup_after_evaluation(self) -> None:
        """Clean up after evaluation."""
        self._metrics = None
        self._motion_failed = None
        self._per_component_failures = {}
        self._component_value_sum = {}
        self._component_value_min = {}
        self._component_value_max = {}
        self._component_step_count = {}
        self._component_manager = None

    def _init_eval_component_buffers(self, num_eval_ids: int) -> None:
        """Initialize per-component failure and value accumulators for this evaluation run."""
        if not self.config.evaluation_components:
            return

        self._motion_failed = torch.zeros(num_eval_ids, dtype=torch.bool, device=self.device)
        self._per_component_failures = {
            name: torch.zeros(num_eval_ids, dtype=torch.bool, device=self.device)
            for name in self.config.evaluation_components.keys()
        }
        self._component_value_sum = {
            name: torch.zeros(num_eval_ids, device=self.device)
            for name in self.config.evaluation_components.keys()
        }
        self._component_value_min = {
            name: torch.full((num_eval_ids,), float('inf'), device=self.device)
            for name in self.config.evaluation_components.keys()
        }
        self._component_value_max = {
            name: torch.full((num_eval_ids,), float('-inf'), device=self.device)
            for name in self.config.evaluation_components.keys()
        }
        self._component_step_count = {
            name: torch.zeros(num_eval_ids, dtype=torch.long, device=self.device)
            for name in self.config.evaluation_components.keys()
        }

        self._component_manager = ComponentManager(self.device)

    def _check_evaluation_failures(
        self,
        active_env_ids: Tensor,
        active_motion_ids: Tensor,
    ) -> None:
        """Check evaluation components and accumulate values/failures for active motions."""
        if self._component_manager is None:
            return

        raw_values = self._component_manager.execute_all(
            self.config.evaluation_components, self.env.context
        )
        failed_buf, component_values, component_failures = combine_evaluation(
            raw_values=raw_values,
            configs=self.config.evaluation_components,
            num_envs=self.agent.num_envs,
            device=self.device,
        )

        # Vectorized update of motion failures
        active_failed = failed_buf[active_env_ids]
        self._motion_failed[active_motion_ids] = self._motion_failed[active_motion_ids] | active_failed

        for name, failures in component_failures.items():
            active_failures = failures[active_env_ids]
            self._per_component_failures[name][active_motion_ids] = (
                self._per_component_failures[name][active_motion_ids] | active_failures
            )

        for name, values in component_values.items():
            active_vals = values[active_env_ids]
            self._component_value_sum[name][active_motion_ids] += active_vals
            self._component_value_min[name][active_motion_ids] = torch.minimum(
                self._component_value_min[name][active_motion_ids], active_vals
            )
            self._component_value_max[name][active_motion_ids] = torch.maximum(
                self._component_value_max[name][active_motion_ids], active_vals
            )
            self._component_step_count[name][active_motion_ids] += 1

    def _create_base_metrics(
        self,
        metric_keys: list,
        num_motions: int,
        motion_num_frames: torch.Tensor,
        max_eval_steps: int,
    ) -> Dict[str, MotionMetrics]:
        """
        Create MotionMetrics objects for a list of keys.

        Args:
            metric_keys: List of metric keys to create
            num_motions: Number of motions to evaluate
            motion_num_frames: Number of frames per motion
            max_eval_steps: Maximum evaluation steps

        Returns:
            Dictionary of MotionMetrics objects
        """
        metrics = {}
        for k in metric_keys:
            metrics[k] = MotionMetrics(
                num_motions, motion_num_frames, max_eval_steps, device=self.device
            )
        return metrics

    def _add_robot_state_metrics(
        self,
        metrics: Dict[str, MotionMetrics],
        num_motions: int,
        motion_num_frames: torch.Tensor,
        max_eval_steps: int,
    ) -> None:
        """
        Add metrics for raw robot state (dof_pos, rigid_body_pos, etc.).
        This is needed for derived metrics like smoothness.

        Args:
            metrics: Existing metrics dict to add to
            num_motions: Number of motions to evaluate
            motion_num_frames: Number of frames per motion
            max_eval_steps: Maximum evaluation steps
        """
        # Default implementation for humanoid robot state
        if not hasattr(self.env, "simulator"):
            return

        try:
            from protomotions.simulator.base_simulator.simulator_state import RobotState

            dummy_state: RobotState = self.env.simulator.get_robot_state()
            shape_mapping = dummy_state.get_shape_mapping(flattened=True)

            for k, shape in shape_mapping.items():
                metrics[k] = MotionMetrics(
                    num_motions,
                    motion_num_frames,
                    max_eval_steps,
                    num_sub_features=shape[0],
                    device=self.device,
                )
        except (AttributeError, KeyError, IndexError) as e:
            log.warning("Could not add robot state metrics: %s", e)

    def _register_plugins(self) -> None:
        """Register metric computation plugins. Override in subclasses."""
        pass

    def _register_smoothness_plugin(
        self, window_sec: float = 0.4, high_jerk_threshold: float = 6500.0
    ) -> bool:
        """
        Convenience method to register smoothness aggregate metric.

        Args:
            window_sec: Window size in seconds for smoothness computation
            high_jerk_threshold: Threshold for classifying high jerk frames

        Returns:
            True if plugin was registered successfully, False otherwise
        """
        try:
            self.metric_plugins.append(
                SmoothnessAggregateMetric(self, window_sec, high_jerk_threshold)
            )
            return True
        except ValueError as e:
            log.warning("Skipping smoothness plugin: %s", e)
            return False

    def _register_action_smoothness_plugin(self) -> bool:
        """
        Convenience method to register action smoothness aggregate metric.

        Measures how much actions change between consecutive timesteps.

        Returns:
            True if plugin was registered successfully, False otherwise
        """
        try:
            self.metric_plugins.append(ActionSmoothnessAggregateMetric(self))
            return True
        except (ValueError, TypeError) as e:
            log.warning("Skipping action smoothness plugin: %s", e)
            return False

    def _compute_additional_metrics(
        self, metrics: Dict[str, MotionMetrics]
    ) -> Dict[str, float]:
        """
        Run all registered metric plugins to compute additional metrics.

        Args:
            metrics: Dictionary of collected MotionMetrics

        Returns:
            Dictionary of additional computed metrics
        """
        additional_metrics = {}
        for plugin in self.metric_plugins:
            try:
                plugin_metrics = plugin.compute(metrics)
                additional_metrics.update(plugin_metrics)
            except Exception as e:
                log.warning("Plugin %s failed: %s", plugin.__class__.__name__, e)
        return additional_metrics

    def _gen_metrics(
        self, metrics: Dict[str, MotionMetrics], keys_to_log: list, prefix: str = "eval"
    ) -> Dict[str, float]:
        """
        Log metrics with mean/max/min aggregations across motions.

        For each metric, computes:
        - mean: average across all per-motion means (overall performance)
        - max: maximum of per-motion means (worst performing motion)
        - min: minimum of per-motion means (best performing motion)

        This gives you 3 separate line plot groups that track over time:
        - {prefix}_mean/{metric}: How well you perform on average
        - {prefix}_max/{metric}: How well you perform on the hardest motion
        - {prefix}_min/{metric}: How well you perform on the easiest motion

        Args:
            metrics: Dictionary of MotionMetrics
            keys_to_log: List of metric keys to log
            prefix: Base prefix for logged metric names (default: "eval")

        Returns:
            Dictionary of logged metrics
        """
        to_log = {}
        for k in keys_to_log:
            if k in metrics:
                to_log[f"{prefix}_mean/{k}"] = metrics[k].mean_mean_reduce().item()
                to_log[f"{prefix}_max/{k}"] = metrics[k].mean_max_reduce().item()
                to_log[f"{prefix}_min/{k}"] = metrics[k].mean_min_reduce().item()
        return to_log

    def _save_list_to_file(
        self, items: list, filename: str, subdirectory: Optional[str] = None
    ) -> None:
        """
        Save a list of items to a text file (one per line).

        Args:
            items: List of items to save
            filename: Name of output file
            subdirectory: Optional subdirectory within root_dir
        """
        if subdirectory:
            output_dir = self.root_dir / subdirectory
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / filename
        else:
            output_path = self.root_dir / filename

        print(f"Saving to: {output_path}")
        with open(output_path, "w") as f:
            for item in items:
                f.write(f"{item}\n")

    def _plot_per_frame_metrics(
        self,
        metrics: Dict[str, MotionMetrics],
        keys_to_plot: Optional[list] = None,
        motion_id: int = 0,
        custom_colors: Optional[Dict[str, str]] = None,
        output_filename: str = "metrics_per_frame_plot.png",
    ) -> None:
        """
        Plot per-frame metrics vs time for a single motion.
        Only plots single-feature metrics (ignores multi-feature metrics).

        Args:
            metrics: Dictionary of MotionMetrics objects
            keys_to_plot: List of keys to plot (None = plot all single-feature metrics)
            motion_id: Which motion to plot (default: 0)
            custom_colors: Optional dict mapping metric keys to colors
            output_filename: Name of output file
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping plotting")
            return

        dt = self.env.dt
        custom_colors = custom_colors or {}

        # Filter to only single-feature metrics
        single_feature_metrics = {}
        valid_frames = {}

        # Determine which keys to plot
        if keys_to_plot is None:
            keys_to_plot = list(metrics.keys())

        for k in keys_to_plot:
            if k in metrics and metrics[k].num_sub_features == 1:
                single_feature_metrics[k] = metrics[k]
                valid_frames[k] = metrics[k].frame_counts[motion_id].item()

        if not single_feature_metrics:
            print("No single-feature metrics found for plotting")
            return

        # Create subplots for each single-feature metric
        num_metrics = len(single_feature_metrics)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 4 * num_metrics))
        if num_metrics == 1:
            axes = [axes]

        for i, k in enumerate(single_feature_metrics.keys()):
            metric = single_feature_metrics[k]
            num_valid_frames = valid_frames[k]

            if num_valid_frames == 0:
                axes[i].text(
                    0.5,
                    0.5,
                    f"No data for {k}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=axes[i].transAxes,
                )
                axes[i].set_title(f"{k}")
                continue

            # Extract data for the single motion (single feature)
            data = metric.data[motion_id, :num_valid_frames, 0].cpu().numpy()
            time_steps = np.arange(num_valid_frames) * dt

            # Use custom color if provided, otherwise matplotlib default
            plot_kwargs = {"label": k, "linewidth": 2}
            if k in custom_colors:
                plot_kwargs["color"] = custom_colors[k]

            axes[i].plot(time_steps, data, **plot_kwargs)
            axes[i].set_xlabel("Time (s)")
            axes[i].set_ylabel(f"{k}")
            axes[i].set_title(f"{k} vs Time")
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()

        plt.tight_layout()

        # Save the plot
        if hasattr(self, "root_dir") and self.root_dir is not None:
            plot_path = self.root_dir / output_filename
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"Per-frame metrics plot saved to: {plot_path}")

        plt.close(fig)
        print("Per-frame metrics plotted successfully")

    def simple_test_policy(self, collect_metrics: bool = False) -> None:
        """
        Simple evaluation loop for interactive testing.

        Runs policy indefinitely, collecting running average of metrics.
        Press Ctrl+C to stop and print summary.

        Args:
            collect_metrics: If True, collect and print average metrics on exit.
        """
        self.agent.eval()
        done_indices = None
        step = 0

        # Running averages for metrics
        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}

        print("Evaluating policy... (Ctrl+C to stop)")
        try:
            while True:
                obs, _ = self.env.reset(done_indices)
                obs = self.agent.add_agent_info_to_obs(obs)
                obs_td = self.agent.obs_dict_to_tensordict(obs)

                model_outs = self.agent.model(obs_td)
                action = model_outs.get("mean_action", model_outs["action"])

                obs, rewards, dones, terminated, extras = self.env.step(action)
                obs = self.agent.add_agent_info_to_obs(obs)
                obs_td = self.agent.obs_dict_to_tensordict(obs)

                # Accumulate metrics
                if collect_metrics and "eval_values" in extras:
                    for k, v in extras["eval_values"].items():
                        val = v.mean().item()
                        metric_sums[k] = metric_sums.get(k, 0.0) + val
                        metric_counts[k] = metric_counts.get(k, 0) + 1

                done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
                step += 1
        except KeyboardInterrupt:
            print(f"\nStopped after {step} steps.")
            if collect_metrics and metric_counts:
                print("Average metrics:")
                for k in sorted(metric_counts.keys()):
                    avg = metric_sums[k] / metric_counts[k]
                    print(f"  {k}: {avg:.4f}")
