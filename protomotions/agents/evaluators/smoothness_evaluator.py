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
from typing import Dict, Optional, Tuple, List
from .metrics import MotionMetrics


class SmoothnessEvaluator:
    """
    Evaluator for motion smoothness metrics like normalized jerk.

    This class computes smoothness metrics from collected motion data,
    particularly using rigid body positions to derive velocity via finite
    differences and then computing normalized jerk.
    """

    def __init__(
        self,
        device: torch.device,
        dt: float = 1.0 / 30.0,
        window_sec: float = 0.4,
        high_jerk_threshold: float = 6500.0,
    ):
        """
        Initialize the smoothness evaluator.

        Args:
            device: Device to perform computations on
            dt: Time step duration in seconds
            window_sec: Default window size in seconds for rolling window computation
            high_jerk_threshold: Threshold for classifying windows as having high jerk
        """
        self.device = device
        self.dt = dt
        self.window_sec = window_sec
        self.high_jerk_threshold = high_jerk_threshold

    def _diff(self, x: torch.Tensor, dt: float) -> torch.Tensor:
        """Compute finite difference with given time step"""
        return (x[1:] - x[:-1]) / dt

    def compute_normalized_jerk_from_pos(
        self,
        rigid_body_pos_metric: MotionMetrics,
        num_bodies: int,
        window_sec: float = 0.4,
        eps: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Compute normalized jerk from rigid body position data using sliding windows.

        Similar to motion_visualizer_smoothness.py, computes normalized jerk over
        rolling windows rather than the entire motion sequence.

        The normalized jerk is computed as:
        NJ = (T^5 * ∫|jerk|^2 dt) / (path_length^2)

        Using T^5 makes the metric dimensionless and FPS-invariant, allowing
        fair comparison across motions sampled at different frame rates.

        Args:
            rigid_body_pos_metric: MotionMetrics containing rigid body positions
                Shape: [num_motions, max_frames, num_bodies*3]
            num_bodies: Number of rigid bodies
            window_sec: Window size in seconds for rolling window computation
            eps: Small epsilon for numerical stability

        Returns:
            per_motion_nj: Mean normalized jerk per motion [num_motions]
            per_body_per_motion_nj: Mean normalized jerk per body per motion [num_motions, num_bodies]
            windowed_nj_per_motion: List of windowed NJ tensors per motion [num_windows, num_bodies]
        """
        data = rigid_body_pos_metric.data  # [num_motions, max_frames, num_bodies*3]
        frame_counts = rigid_body_pos_metric.frame_counts  # [num_motions]
        num_motions = data.shape[0]

        # Calculate window size in frames (minimum 4 for jerk computation)
        window_frames = max(4, int(round(window_sec / self.dt)))

        # Reshape to [num_motions, max_frames, num_bodies, 3]
        pos = data.view(num_motions, -1, num_bodies, 3)

        per_motion_nj = torch.zeros(num_motions, device=self.device)
        per_body_per_motion_nj = torch.zeros(
            num_motions, num_bodies, device=self.device
        )
        windowed_nj_per_motion = []

        for motion_idx in range(num_motions):
            valid_frames = frame_counts[motion_idx].item()
            if (
                valid_frames < window_frames
            ):  # Need at least window_frames for computation
                # Add empty tensor for motions with insufficient data
                windowed_nj_per_motion.append(
                    torch.empty(0, num_bodies, device=self.device)
                )
                continue

            # Extract valid frames for this motion: [T, num_bodies, 3]
            pos_motion = pos[motion_idx, :valid_frames]

            # Compute windowed normalized jerk efficiently using unfold
            window_nj = self._compute_windowed_normalized_jerk(
                pos_motion, window_frames, eps
            )

            # Store windowed NJ values for this motion
            windowed_nj_per_motion.append(window_nj)

            # window_nj has shape [num_windows, num_bodies]
            # Take mean across windows to get per-body average
            if window_nj.numel() > 0:
                per_body_per_motion_nj[motion_idx] = window_nj.mean(dim=0)
                per_motion_nj[motion_idx] = per_body_per_motion_nj[motion_idx].mean()

        return per_motion_nj, per_body_per_motion_nj, windowed_nj_per_motion

    def _compute_high_jerk_frame_percentage(
        self, windowed_nj: torch.Tensor, threshold: Optional[float] = None
    ) -> float:
        """
        Compute the percentage of windows/frames where at least one body has high jerk.

        Args:
            windowed_nj: Normalized jerk values [num_windows, num_bodies]
            threshold: Threshold for high jerk (uses default if None)

        Returns:
            Percentage of windows with at least one body exceeding threshold
        """
        if windowed_nj.numel() == 0:
            return 0.0

        # Use provided threshold or fall back to default
        thresh = threshold if threshold is not None else self.high_jerk_threshold

        # Check if any body in each window exceeds threshold: [num_windows]
        high_jerk_windows = (windowed_nj > thresh).any(dim=1)

        # Calculate percentage of windows with high jerk
        high_jerk_percentage = high_jerk_windows.float().mean().item() * 100.0

        return high_jerk_percentage

    def _compute_windowed_normalized_jerk(
        self, pos_motion: torch.Tensor, window_frames: int, eps: float = 0.1
    ) -> torch.Tensor:
        """
        Efficiently compute normalized jerk for sliding windows using fully vectorized operations.

        Args:
            pos_motion: Position data [T, num_bodies, 3]
            window_frames: Size of sliding window
            eps: Small epsilon for numerical stability

        Returns:
            torch.Tensor: Normalized jerk per window per body [num_windows, num_bodies]
        """
        T, num_bodies, _ = pos_motion.shape
        num_windows = T - window_frames + 1

        if num_windows <= 0 or window_frames < 4:
            return torch.empty(0, num_bodies, device=self.device)

        # Create sliding windows using unfold: [num_windows, window_frames, num_bodies, 3]
        # We need to permute to get the right dimensions for unfold
        pos_for_unfold = pos_motion.permute(1, 2, 0)  # [num_bodies, 3, T]
        pos_for_unfold = pos_for_unfold.contiguous().view(
            num_bodies * 3, T
        )  # [num_bodies*3, T]

        # Create sliding windows: [num_bodies*3, num_windows, window_frames]
        windowed_pos = pos_for_unfold.unfold(1, window_frames, 1)

        # Reshape back: [num_bodies, 3, num_windows, window_frames]
        windowed_pos = windowed_pos.view(num_bodies, 3, num_windows, window_frames)

        # Transpose to: [num_windows, window_frames, num_bodies, 3]
        windowed_pos = windowed_pos.permute(2, 3, 0, 1)

        # Vectorized computation of derivatives for all windows at once
        # windowed_pos: [num_windows, window_frames, num_bodies, 3]

        # Compute velocity for all windows: [num_windows, window_frames-1, num_bodies, 3]
        vel_all = self._diff_batch(windowed_pos, self.dt)

        # Compute acceleration for all windows: [num_windows, window_frames-2, num_bodies, 3]
        acc_all = self._diff_batch(vel_all, self.dt)

        # Compute jerk for all windows: [num_windows, window_frames-3, num_bodies, 3]
        jerk_all = self._diff_batch(acc_all, self.dt)

        # Now compute normalized jerk for all windows in parallel
        T_tot = (window_frames - 1) * self.dt

        # Speed magnitude for all windows: [num_windows, window_frames-1, num_bodies]
        speed_all = torch.linalg.norm(vel_all, dim=-1)

        # Jerk magnitude squared for all windows: [num_windows, window_frames-3, num_bodies]
        jerk_norm_squared_all = torch.linalg.norm(jerk_all, dim=-1) ** 2

        # Path length per body per window: [num_windows, num_bodies]
        path_length_all = (speed_all * self.dt).sum(dim=1).clamp_min(eps)

        # Integrated squared jerk per body per window: [num_windows, num_bodies]
        integrated_jerk_squared_all = (jerk_norm_squared_all * self.dt).sum(dim=1)

        # Normalized jerk per body per window: [num_windows, num_bodies]
        # Using T^5 (not T^3) for dimensionless, FPS-invariant normalization
        nj_all = (T_tot**5 * integrated_jerk_squared_all) / (path_length_all**2 + eps)

        return nj_all

    def _diff_batch(self, x: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Compute finite difference for batched data along the time dimension (dimension 1).

        Args:
            x: Input tensor [num_windows, T, num_bodies, 3]
            dt: Time step

        Returns:
            Tensor with T-1 frames along the time dimension [num_windows, T-1, num_bodies, 3]
        """
        # Time dimension is dimension 1 for windowed_pos
        return (x[:, 1:, :, :] - x[:, :-1, :, :]) / dt

    def compute_smoothness_metrics(
        self,
        metrics: Dict[str, MotionMetrics],
        num_bodies: int,
        window_sec: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute smoothness metrics from collected motion data using sliding windows.

        Args:
            metrics: Dictionary of collected MotionMetrics
            num_bodies: Number of rigid bodies in the robot
            window_sec: Window size in seconds (uses default if None)

        Returns:
            Dictionary of smoothness metrics for logging
        """
        smoothness_metrics = {}

        if "rigid_body_pos" not in metrics:
            return smoothness_metrics

        # Use provided window_sec or fall back to default
        window_size = window_sec if window_sec is not None else self.window_sec

        try:
            # Compute normalized jerk from rigid body positions using sliding windows
            per_motion_nj, per_body_per_motion_nj, windowed_nj_per_motion = (
                self.compute_normalized_jerk_from_pos(
                    metrics["rigid_body_pos"], num_bodies, window_sec=window_size
                )
            )

            # Filter out motions with insufficient data (NJ = 0)
            valid_motions_mask = per_motion_nj > 0

            if valid_motions_mask.any():
                valid_nj = per_motion_nj[valid_motions_mask]

                smoothness_metrics.update(
                    {
                        "normalized_jerk_mean": valid_nj.mean().item(),
                        # "normalized_jerk_max": valid_nj.max().item(),
                        # "normalized_jerk_min": valid_nj.min().item(),
                        # "normalized_jerk_std": valid_nj.std().item(),
                    }
                )

                # # Compute body-specific statistics
                # valid_body_nj = per_body_per_motion_nj[valid_motions_mask]  # [valid_motions, num_bodies]
                # body_mean_nj = valid_body_nj.mean(dim=0)  # [num_bodies] - mean NJ per body across motions

                # smoothness_metrics.update({
                #     "normalized_jerk_body_mean": body_mean_nj.mean().item(),  # Overall mean across bodies
                #     "normalized_jerk_body_max": body_mean_nj.max().item(),    # Max body mean NJ
                #     "normalized_jerk_body_min": body_mean_nj.min().item(),    # Min body mean NJ
                # })

                # Compute high jerk frame percentage across all motions
                high_jerk_percentages = []
                for windowed_nj in windowed_nj_per_motion:
                    if windowed_nj.numel() > 0:  # Only process motions with valid data
                        high_jerk_pct = self._compute_high_jerk_frame_percentage(
                            windowed_nj
                        )
                        high_jerk_percentages.append(high_jerk_pct)

                if high_jerk_percentages:
                    smoothness_metrics.update(
                        {
                            "high_jerk_frame_percentage_mean": sum(
                                high_jerk_percentages
                            )
                            / len(high_jerk_percentages),
                            # "high_jerk_frame_percentage_max": max(high_jerk_percentages),
                            # "high_jerk_frame_percentage_min": min(high_jerk_percentages),
                        }
                    )
                else:
                    smoothness_metrics.update(
                        {
                            "high_jerk_frame_percentage_mean": 0.0,
                            # "high_jerk_frame_percentage_max": 0.0,
                            # "high_jerk_frame_percentage_min": 0.0,
                        }
                    )

            else:
                # No valid motions for jitter computation
                smoothness_metrics.update(
                    {
                        "normalized_jerk_mean": 0.0,
                        # "normalized_jerk_max": 0.0,
                        # "normalized_jerk_min": 0.0,
                        # "normalized_jerk_std": 0.0,
                        # "normalized_jerk_body_mean": 0.0,
                        # "normalized_jerk_body_max": 0.0,
                        # "normalized_jerk_body_min": 0.0,
                        "high_jerk_frame_percentage_mean": 0.0,
                        # "high_jerk_frame_percentage_max": 0.0,
                        # "high_jerk_frame_percentage_min": 0.0,
                    }
                )

        except Exception as e:
            print(f"Warning: Failed to compute normalized jerk: {e}")

        return smoothness_metrics


if __name__ == "__main__":
    import torch
    import math
    from protomotions.agents.evaluators.smoothness_evaluator import SmoothnessEvaluator
    from protomotions.agents.evaluators.metrics import MotionMetrics

    device = torch.device("cpu")
    # Use realistic values for testing
    evaluator = SmoothnessEvaluator(
        device=device,
        dt=1.0 / 30.0,  # 30 FPS
        window_sec=0.4,  # 0.4s window
        high_jerk_threshold=6500.0,  # FPS-invariant threshold (was 50000 with T^3)
    )

    # Create diverse test data
    num_motions, max_frames, num_bodies = 3, 40, 4
    motion_lens = torch.tensor([40, 35, 30])

    pos_data = torch.zeros(num_motions, max_frames, num_bodies * 3)

    # Motion 0: Very smooth motion
    for t in range(40):
        pos_data[0, t, :] = t * 0.02  # Linear motion

    # Motion 1: Mixed smooth and jerky
    for t in range(35):
        if t < 20:
            pos_data[1, t, :] = t * 0.02  # Smooth start
        else:
            pos_data[1, t, :] = 0.4 + math.sin((t - 20) * 4.0) * 0.3  # Jerky end

    # Motion 2: Very jerky throughout
    for t in range(30):
        pos_data[2, t, :] = math.sin(t * 2.0) * 0.5 + math.cos(t * 3.0) * 0.3

    # Create MotionMetrics object
    metrics = MotionMetrics(
        num_motions, motion_lens, max_frames, num_bodies * 3, device
    )
    metrics.data = pos_data
    metrics.frame_counts = motion_lens

    # Compute all smoothness metrics
    smoothness_metrics = evaluator.compute_smoothness_metrics(
        {"rigid_body_pos": metrics}, num_bodies
    )

    print("Evaluator Configuration:")
    print(f"  Window size: {evaluator.window_sec}s")
    print(f"  High jerk threshold: {evaluator.high_jerk_threshold}")
    print(f"  Time step: {evaluator.dt:.4f}s ({1/evaluator.dt:.1f} FPS)")
    print()

    print("📊 Complete Smoothness Metrics:")
    print()

    print("Mean-Based Metrics:")
    for key in [
        "normalized_jerk_mean",
        "normalized_jerk_max",
        "normalized_jerk_min",
        "normalized_jerk_std",
    ]:
        if key in smoothness_metrics:
            print(f"  {key:<25}: {smoothness_metrics[key]:>12.2f}")

    print()
    print("Body-Specific Metrics:")
    for key in [
        "normalized_jerk_body_mean",
        "normalized_jerk_body_max",
        "normalized_jerk_body_min",
    ]:
        if key in smoothness_metrics:
            print(f"  {key:<25}: {smoothness_metrics[key]:>12.2f}")

    print()
    print("High Jerk Frame Metrics (NEW):")
    for key in [
        "high_jerk_frame_percentage_mean",
        "high_jerk_frame_percentage_max",
        "high_jerk_frame_percentage_min",
    ]:
        if key in smoothness_metrics:
            print(f"  {key:<25}: {smoothness_metrics[key]:>12.2f}%")

    print()
