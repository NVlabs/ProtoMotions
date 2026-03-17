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
"""Aggregate metrics for post-hoc trajectory analysis.

This module contains metrics computed over accumulated MotionMetrics trajectories
after an evaluation episode completes, as opposed to per-step evaluation components
(MdpComponents) which run during the episode with threshold-based failure detection.

Key Classes:
    - AggregateMetric: Base class for post-hoc metrics
    - SmoothnessAggregateMetric: Computes normalized jerk and high-jerk frame percentage
    - ActionSmoothnessAggregateMetric: Computes action rate of change metrics
"""

from typing import Dict
from protomotions.agents.evaluators.metrics import MotionMetrics
from protomotions.agents.evaluators.smoothness_calculator import SmoothnessCalculator


class AggregateMetric:
    """Base class for metrics computed post-hoc over accumulated MotionMetrics trajectories.
    
    Unlike evaluation_components (MdpComponents) which run per-step with 
    threshold-based failure detection, AggregateMetrics run after the full 
    episode to compute summary statistics like smoothness or jerk.
    
    Subclasses must implement the compute() method.
    """
    
    def compute(self, metrics: Dict[str, MotionMetrics]) -> Dict[str, float]:
        """Compute aggregate metrics from collected motion data.
        
        Args:
            metrics: Dictionary of MotionMetrics objects containing trajectory data
        
        Returns:
            Dictionary of scalar metrics with "eval/" prefix for logging
        """
        raise NotImplementedError


class SmoothnessAggregateMetric(AggregateMetric):
    """Aggregate metric for computing motion smoothness from rigid body trajectories.
    
    Computes normalized jerk and high-jerk frame percentage over sliding windows.
    """

    def __init__(
        self, evaluator, window_sec: float = 0.4, high_jerk_threshold: float = 6500.0
    ):
        """Initialize the smoothness aggregate metric.

        Args:
            evaluator: The parent evaluator instance
            window_sec: Window size in seconds for smoothness computation
            high_jerk_threshold: Threshold for classifying high jerk frames
        """
        self.smoothness_calculator = SmoothnessCalculator(
            device=evaluator.device,
            dt=evaluator.env.dt,
            window_sec=window_sec,
            high_jerk_threshold=high_jerk_threshold,
        )
        self.num_bodies = evaluator.env.robot_config.kinematic_info.num_bodies

    def compute(self, metrics: Dict[str, MotionMetrics]) -> Dict[str, float]:
        """Compute smoothness metrics from collected motion data.

        Args:
            metrics: Dictionary of MotionMetrics

        Returns:
            Dictionary of smoothness metrics with "eval/" prefix
        """
        smoothness_metrics = self.smoothness_calculator.compute_smoothness_metrics(
            metrics, self.num_bodies
        )

        # Add logging for each smoothness metric
        result = {}
        for k, v in smoothness_metrics.items():
            print(f"Smoothness metric: {k}, value: {v}")
            result[f"eval/{k}"] = v

        return result


class ActionSmoothnessAggregateMetric(AggregateMetric):
    """Aggregate metric for computing action smoothness.
    
    Measures how much actions change between consecutive timesteps.
    High action deltas indicate jerky/unstable control.
    """

    def __init__(self, evaluator, dt: float = None):
        """Initialize the action smoothness aggregate metric.

        Args:
            evaluator: The parent evaluator instance
            dt: Simulation timestep (defaults to env.dt)
        """
        self.dt = dt if dt is not None else evaluator.env.dt
        self.device = evaluator.device

    def compute(self, metrics: Dict[str, MotionMetrics]) -> Dict[str, float]:
        """Compute action smoothness metrics from collected action data.

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

            print(
                f"Action smoothness: mean_delta={mean_delta:.4f} rad ({mean_delta * 180 / 3.14159:.2f}°), "
                f"max_delta={mean_max_delta:.4f} rad ({mean_max_delta * 180 / 3.14159:.2f}°)"
            )

        return result
