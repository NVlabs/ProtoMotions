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
"""Base evaluator for agent evaluation and metrics computation.

This module provides the base evaluation infrastructure for computing performance
metrics during training and evaluation. Evaluators run periodic assessments of
agent performance and compute task-specific metrics.

Key Classes:
    - BaseEvaluator: Base class for all evaluators
    - SmoothnessMetricPlugin: Plugin for computing motion smoothness metrics

Key Features:
    - Periodic evaluation during training
    - Motion quality metrics computation
    - Episode statistics aggregation
    - Smoothness and jerk analysis
    - Distributed evaluation support
"""

import torch
from typing import Dict, Optional, Tuple, Any
from lightning.fabric import Fabric
from protomotions.agents.evaluators.metrics import MotionMetrics
from protomotions.agents.evaluators.smoothness_evaluator import SmoothnessEvaluator
from protomotions.envs.base_env.env import BaseEnv
from protomotions.agents.evaluators.config import EvaluatorConfig


class SmoothnessMetricPlugin:
    """Plugin for computing smoothness metrics from motion data."""

    def __init__(
        self, evaluator, window_sec: float = 0.4, high_jerk_threshold: float = 6500.0
    ):
        """
        Initialize the smoothness metric plugin.

        Args:
            evaluator: The parent evaluator instance
            window_sec: Window size in seconds for smoothness computation
            high_jerk_threshold: Threshold for classifying high jerk frames
        """
        self.smoothness_evaluator = SmoothnessEvaluator(
            device=evaluator.device,
            dt=evaluator.env.dt,
            window_sec=window_sec,
            high_jerk_threshold=high_jerk_threshold,
        )
        self.num_bodies = evaluator.env.robot_config.kinematic_info.num_bodies

    def compute(self, metrics: Dict[str, MotionMetrics]) -> Dict[str, float]:
        """
        Compute smoothness metrics from collected motion data.

        Args:
            metrics: Dictionary of MotionMetrics

        Returns:
            Dictionary of smoothness metrics with "eval/" prefix
        """
        smoothness_metrics = self.smoothness_evaluator.compute_smoothness_metrics(
            metrics, self.num_bodies
        )

        # Add logging for each smoothness metric
        result = {}
        for k, v in smoothness_metrics.items():
            print(f"Smoothness metric: {k}, value: {v}")
            result[f"eval/{k}"] = v

        return result


class ActionSmoothnessMetricPlugin:
    """Plugin for computing action smoothness metrics.
    
    Measures how much actions change between consecutive timesteps.
    High action deltas indicate jerky/unstable control.
    """

    def __init__(self, evaluator, dt: float = None):
        """
        Initialize the action smoothness metric plugin.

        Args:
            evaluator: The parent evaluator instance
            dt: Simulation timestep (defaults to env.dt)
        """
        self.dt = dt if dt is not None else evaluator.env.dt
        self.device = evaluator.device

    def compute(self, metrics: Dict[str, MotionMetrics]) -> Dict[str, float]:
        """
        Compute action smoothness metrics from collected action data.

        Metrics computed:
        - action_delta_mean: Mean absolute action change per step (rad)
        - action_delta_max: Max absolute action change per step across all joints (rad)
        - action_rate_mean: Mean action rate of change (rad/s)

        Args:
            metrics: Dictionary of MotionMetrics (must contain "actions")

        Returns:
            Dictionary of action smoothness metrics with "eval/" prefix
        """
        if "actions" not in metrics:
            return {}

        actions_data = metrics["actions"].data  # [num_motions, max_frames, num_dofs]
        motion_lens = metrics["actions"].motion_lens  # [num_motions]
        
        result = {}
        all_deltas = []
        all_max_deltas = []
        
        num_motions = actions_data.shape[0]
        for m in range(num_motions):
            n_frames = int(motion_lens[m].item())
            if n_frames < 2:
                continue
            
            actions_m = actions_data[m, :n_frames]  # [n_frames, num_dofs]
            # Compute action deltas between consecutive frames
            deltas = (actions_m[1:] - actions_m[:-1]).abs()  # [n_frames-1, num_dofs]
            
            # Mean delta across all joints and frames
            all_deltas.append(deltas.mean().item())
            # Max delta per frame, then mean across frames
            all_max_deltas.append(deltas.max(dim=-1)[0].mean().item())
        
        if all_deltas:
            mean_delta = sum(all_deltas) / len(all_deltas)
            mean_max_delta = sum(all_max_deltas) / len(all_max_deltas)
            
            result["eval/action_delta_mean_rad"] = mean_delta
            result["eval/action_delta_max_rad"] = mean_max_delta
            result["eval/action_rate_mean_rad_s"] = mean_delta / self.dt
            
            # Convert to degrees for readability
            result["eval/action_delta_mean_deg"] = mean_delta * 180 / 3.14159
            result["eval/action_delta_max_deg"] = mean_max_delta * 180 / 3.14159
            
            print(f"Action smoothness: mean_delta={mean_delta:.4f} rad ({mean_delta * 180 / 3.14159:.2f}°), "
                  f"max_delta={mean_max_delta:.4f} rad ({mean_max_delta * 180 / 3.14159:.2f}°)")
        
        return result


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

        # Plugin system for additional metrics
        self.metric_plugins = []
        self._register_plugins()

        # Counter for tracking evaluation calls
        self.eval_count = 0

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
        self.agent.eval()

        # Initialize metrics and prepare evaluation context
        metrics = self.initialize_eval()
        if not metrics:
            return {}, 0

        # Run evaluation
        self.run_evaluation(metrics)

        # Process evaluation results
        evaluation_log, evaluated_score = self.process_eval_results(metrics)

        # Cleanup after evaluation
        self.cleanup_after_evaluation()

        # Increment eval counter
        self.eval_count += 1

        return evaluation_log, evaluated_score

    def initialize_eval(self) -> Tuple[Dict, Dict]:
        """
        Initialize metrics dictionary with required keys.
        Prepare the evaluation context.

        Returns:
            Tuple containing metrics dict and evaluation context dict
        """
        return {}

    def run_evaluation(self, metrics: Dict) -> None:
        """
        Run the evaluation process and collect metrics.

        Args:
            metrics: Dictionary to collect evaluation metrics
        """
        raise NotImplementedError("Run evaluation not implemented for base evaluator.")

    def process_eval_results(
        self, metrics: Dict, eval_context: Dict
    ) -> Tuple[Dict, Optional[float]]:
        """
        Process collected metrics and prepare for logging.

        Args:
            metrics: Dictionary of collected metrics
            eval_context: Dictionary containing evaluation context

        Returns:
            Tuple containing:
                - Dict of processed metrics for logging
                - Optional score value for determining best model
        """
        return {}, None

    def cleanup_after_evaluation(self) -> None:
        """Clean up after evaluation (reset env state, etc.)"""
        pass

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
        except Exception as e:
            print(f"Warning: Could not add robot state metrics: {e}")

    def _register_plugins(self) -> None:
        """Register metric computation plugins. Override in subclasses."""
        pass

    def _register_smoothness_plugin(
        self, window_sec: float = 0.4, high_jerk_threshold: float = 6500.0
    ) -> bool:
        """
        Convenience method to register smoothness metric plugin.

        Args:
            window_sec: Window size in seconds for smoothness computation
            high_jerk_threshold: Threshold for classifying high jerk frames

        Returns:
            True if plugin was registered successfully, False otherwise
        """
        try:
            self.metric_plugins.append(
                SmoothnessMetricPlugin(self, window_sec, high_jerk_threshold)
            )
            return True
        except ValueError as e:
            print(f"Skipping smoothness plugin: {e}")
            return False

    def _register_action_smoothness_plugin(self) -> bool:
        """
        Convenience method to register action smoothness metric plugin.

        Measures how much actions change between consecutive timesteps.

        Returns:
            True if plugin was registered successfully, False otherwise
        """
        try:
            self.metric_plugins.append(ActionSmoothnessMetricPlugin(self))
            return True
        except Exception as e:
            print(f"Skipping action smoothness plugin: {e}")
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
                print(f"Warning: Plugin {plugin.__class__.__name__} failed: {e}")
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
            import numpy as np
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
        Simple evaluation loop for testing the policy.

        Args:
            collect_metrics: whether to collect metrics during evaluation
        """
        self.agent.eval()
        done_indices = None  # Force reset on first entry
        step = 0
        print("Evaluating policy...")

        try:
            while True:
                obs, _ = self.env.reset(done_indices)
                obs = self.agent.add_agent_info_to_obs(obs)
                obs_td = self.agent.obs_dict_to_tensordict(obs)

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

                done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
                step += 1
        except KeyboardInterrupt:
            print("\nEvaluation interrupted by Ctrl+C, exiting...")

        return None
