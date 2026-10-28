# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import torch
from typing import Dict, Optional, Tuple, List
from .metrics import MotionMetrics

log = logging.getLogger(__name__)


class SmoothnessCalculator:
    """
    Calculator for motion smoothness metrics like normalized jerk.

    This class computes smoothness metrics from collected motion data,
    particularly using rigid body positions to derive velocity via finite
    differences and then computing normalized jerk.
    """

    def __init__(
        self,
        device: torch.device,
        dt: float,
        window_sec: float = 0.4,
        high_jerk_threshold: float = 6500.0,
    ):
        """
        Initialize the smoothness evaluator.

        Args:
            device: Device to perform computations on
            dt: Time step duration in seconds (must be explicitly provided)
            window_sec: Default window size in seconds for rolling window computation
            high_jerk_threshold: Threshold for classifying windows as having high jerk
        """
        self.device = device
        self.dt = dt
        self.window_sec = window_sec
        self.high_jerk_threshold = high_jerk_threshold

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
            per_motion_nj, _, windowed_nj_per_motion = (
                self.compute_normalized_jerk_from_pos(
                    metrics["rigid_body_pos"], num_bodies, window_sec=window_size
                )
            )

            # Filter out motions with insufficient data (NJ = 0)
            valid_motions_mask = per_motion_nj > 0

            if valid_motions_mask.any():
                valid_nj = per_motion_nj[valid_motions_mask]

                smoothness_metrics["normalized_jerk_mean"] = valid_nj.mean().item()

                # Compute high jerk frame percentage across all motions
                high_jerk_percentages = []
                for windowed_nj in windowed_nj_per_motion:
                    if windowed_nj.numel() > 0:  # Only process motions with valid data
                        high_jerk_pct = self._compute_high_jerk_frame_percentage(
                            windowed_nj
                        )
                        high_jerk_percentages.append(high_jerk_pct)

                smoothness_metrics["high_jerk_frame_percentage_mean"] = (
                    sum(high_jerk_percentages) / len(high_jerk_percentages)
                    if high_jerk_percentages
                    else 0.0
                )
            else:
                # No valid motions for jitter computation
                smoothness_metrics.update(
                    {
                        "normalized_jerk_mean": 0.0,
                        "high_jerk_frame_percentage_mean": 0.0,
                    }
                )

        except (RuntimeError, ValueError, IndexError) as e:
            log.warning("Failed to compute normalized jerk: %s", e)

        return smoothness_metrics
