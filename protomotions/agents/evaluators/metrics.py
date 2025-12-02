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
from typing import Optional, Callable


class MotionMetrics:
    """
    Store and compute metrics for motion data.

    Stores raw data in the shape [num_motions, max_motion_len, num_sub_features] and
    supports basic reduction operations for computing final metrics.
    """

    def __init__(
        self,
        num_motions: int,
        motion_lens: torch.Tensor,
        max_motion_len: int,
        num_sub_features: int = 1,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the metrics tracker.

        Args:
            num_motions: Number of motions to track
            motion_lens: Number of frames of each motion sequence
            max_motion_len: conservative max number of frames allocated for data storage
                for shape consistency across different GPUs when aggregating
            num_sub_features: Number of sub-features per data point (default: 1)
            device: Device to store the tensors on
            dtype: Data type for the tensors
        """
        self.num_motions = num_motions
        self.num_sub_features = num_sub_features
        self.device = device
        self.dtype = dtype
        self.motion_lens = motion_lens
        self.max_motion_len = max_motion_len

        # Raw data storage
        self.data = torch.zeros(
            (num_motions, self.max_motion_len, num_sub_features),
            device=device,
            dtype=dtype,
        )

        # Counters to track number of frames per motion
        self.frame_counts = torch.zeros(num_motions, device=device, dtype=torch.long)

    def update(
        self,
        motion_ids: torch.Tensor,
        values: torch.Tensor,
        frame_indices: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Update the metrics data for specified motions.

        Args:
            motion_ids: Tensor of motion IDs to update [batch_size]
            values: Tensor of values to update [batch_size, num_sub_features]
            frame_indices: Optional tensor of frame indices [batch_size]
                           If None, will use the current count for each motion
        """
        if values.ndim == 1:
            values = values.unsqueeze(1)
        assert motion_ids.shape[0] == values.shape[0]

        # assert motion_ids being non-duplicated
        assert torch.unique(motion_ids).shape[0] == motion_ids.shape[0]

        if frame_indices is None:
            # Use current counts as frame indices
            frame_indices = self.frame_counts[motion_ids]
        assert frame_indices.shape[0] == values.shape[0]

        # Update the data using batched operations with per-motion length checks
        valid_mask = frame_indices < self.motion_lens[motion_ids]
        if valid_mask.any():
            valid_motion_ids = motion_ids[valid_mask]
            valid_frame_indices = frame_indices[valid_mask]
            valid_values = values[valid_mask]

            # Use advanced indexing to update only valid entries
            self.data[valid_motion_ids, valid_frame_indices] = valid_values

            # Increment frame counts
            self.frame_counts[valid_motion_ids] += 1

    def get_unfilled_mask(
        self,
    ) -> torch.Tensor:
        """
        Get a mask of the unfilled values in the data.
        """
        # Create indices matrix and mask for values beyond frame count
        indices = torch.arange(self.max_motion_len, device=self.device).unsqueeze(
            0
        )  # [1, max_motion_len]
        frame_counts = self.frame_counts.unsqueeze(1)  # [num_motions, 1]
        mask = indices >= frame_counts  # [num_motions, max_motion_len]

        # Expand mask to cover sub-features dimension
        mask = mask.unsqueeze(-1).expand(
            -1, -1, self.num_sub_features
        )  # [num_motions, max_motion_len, num_sub_features]

        return mask

    def max_reduce_each_motion(
        self,
        with_frame: bool = False,
    ) -> torch.Tensor:
        """
        Reduce the data by taking the max of each motion.
        """
        mask = self.get_unfilled_mask()

        # Apply the mask to set values beyond frame count to -inf
        data = self.data.masked_fill(mask, float("-inf"))

        # take the max of each motion
        max_values, max_frames = data.max(dim=1)  # [num_motions, num_sub_features]
        if self.num_sub_features == 1:
            max_values = max_values[:, 0]
        if with_frame:
            return max_values, max_frames
        else:
            return max_values

    def min_reduce_each_motion(
        self,
    ) -> torch.Tensor:
        """
        Reduce the data by taking the min of each motion.
        """
        mask = self.get_unfilled_mask()

        # Apply the mask to set values beyond frame count to inf
        data = self.data.masked_fill(mask, float("inf"))

        # take the min of each motion
        min_values = data.min(dim=1).values  # [num_motions, num_sub_features]

        if self.num_sub_features == 1:
            return min_values[:, 0]
        else:
            return min_values

    def mean_reduce_each_motion(
        self,
    ) -> torch.Tensor:
        """
        Reduce the data by taking the mean of each motion.
        """
        mask = self.get_unfilled_mask()

        # Apply the mask to set values beyond frame count to 0 for summation
        data = self.data.masked_fill(mask, 0)

        # Sum the valid values for each motion
        sum_values = data.sum(dim=1)  # [num_motions, num_sub_features]

        # Get the actual number of frames for each motion
        frame_counts = self.frame_counts.unsqueeze(-1).clamp(
            min=1
        )  # [num_motions, 1], clamp to avoid division by zero

        # Calculate the mean by dividing the sum by the actual frame count
        mean_values = sum_values / frame_counts  # [num_motions, num_sub_features]

        # Handle motions with zero frames explicitly, setting their mean to 0 (or NaN if preferred)
        zero_frame_mask = (self.frame_counts == 0).unsqueeze(-1)
        mean_values = mean_values.masked_fill(zero_frame_mask, 0.0)

        if self.num_sub_features == 1:
            return mean_values[:, 0]
        else:
            return mean_values

    def ops_mean_reduce(
        self,
        op: Callable,
    ) -> torch.Tensor:
        """
        first reduce the data by taking the op of each motion, then mean reduce across motions.
        """
        # Check if op is a bound method of this instance
        if hasattr(op, "__self__") and op.__self__ is self:
            op_values = op()  # Call bound method directly
        else:
            op_values = op(self)  # Call external function with self as argument

        op_values_valid = op_values[self.frame_counts > 0]
        # take the mean across num of motions
        mean_values = op_values_valid.mean(dim=0)  # [num_sub_features] or scalar
        return mean_values

    def max_mean_reduce(
        self,
    ) -> torch.Tensor:
        return self.ops_mean_reduce(self.max_reduce_each_motion)

    def min_mean_reduce(
        self,
    ) -> torch.Tensor:
        return self.ops_mean_reduce(self.min_reduce_each_motion)

    def mean_mean_reduce(
        self,
    ) -> torch.Tensor:
        return self.ops_mean_reduce(self.mean_reduce_each_motion)

    def mean_max_reduce(
        self,
    ) -> torch.Tensor:
        """
        First reduce each motion by taking the mean over valid frames,
        then take the max across all motions.

        Returns:
            torch.Tensor: Maximum of the per-motion means (worst performing motion)
        """
        mean_values = (
            self.mean_reduce_each_motion()
        )  # [num_motions] or [num_motions, num_sub_features]

        # Only consider motions with valid frames
        valid_mask = self.frame_counts > 0
        if not valid_mask.any():
            # No valid motions, return zeros with appropriate shape
            if self.num_sub_features == 1:
                return torch.tensor(0.0, device=self.device, dtype=self.dtype)
            else:
                return torch.zeros(
                    self.num_sub_features, device=self.device, dtype=self.dtype
                )

        mean_values_valid = mean_values[valid_mask]
        max_value = (
            mean_values_valid.max(dim=0).values
            if mean_values_valid.ndim > 1
            else mean_values_valid.max()
        )

        return max_value

    def mean_min_reduce(
        self,
    ) -> torch.Tensor:
        """
        First reduce each motion by taking the mean over valid frames,
        then take the min across all motions.

        Returns:
            torch.Tensor: Minimum of the per-motion means (best performing motion)
        """
        mean_values = (
            self.mean_reduce_each_motion()
        )  # [num_motions] or [num_motions, num_sub_features]

        # Only consider motions with valid frames
        valid_mask = self.frame_counts > 0
        if not valid_mask.any():
            # No valid motions, return zeros with appropriate shape
            if self.num_sub_features == 1:
                return torch.tensor(0.0, device=self.device, dtype=self.dtype)
            else:
                return torch.zeros(
                    self.num_sub_features, device=self.device, dtype=self.dtype
                )

        mean_values_valid = mean_values[valid_mask]
        min_value = (
            mean_values_valid.min(dim=0).values
            if mean_values_valid.ndim > 1
            else mean_values_valid.min()
        )

        return min_value

    def compute_finite_difference_jitter_reduce_each_motion(
        self,
        num_bodies: int,
        aggregate_method: str = "mean",
        order: int = 2,
        field_description: str = "data",
    ) -> torch.Tensor:
        """
        Generic method to compute jitter using finite differences of specified order.
        Output is padded to match input length (padded with zeros at the beginning).

        Args:
            num_bodies: Number of rigid bodies (to reshape the flattened data)
            aggregate_method: How to aggregate across bodies ("mean", "max", "sum")
            order: Order of finite differences (1 for velocity-like, 2 for acceleration-like)
            field_description: Description of the field for error messages

        Returns:
            torch.Tensor: Jitter values with shape [num_motions, max_motion_len] (same as input)
        """
        assert (
            self.num_sub_features == num_bodies * 3
        ), f"Expected num_sub_features={num_bodies * 3}, got {self.num_sub_features}"
        assert order in [
            1,
            2,
        ], f"Only 1st and 2nd order finite differences supported, got {order}"

        # Get the mask for valid data
        mask = (
            self.get_unfilled_mask()
        )  # [num_motions, max_motion_len, num_sub_features]

        # Apply mask to data (set invalid entries to 0)
        data = self.data.masked_fill(
            mask, 0.0
        )  # [num_motions, max_motion_len, num_bodies*3]

        # Reshape to separate bodies: [num_motions, max_motion_len, num_bodies, 3]
        data_reshaped = data.view(self.num_motions, self.max_motion_len, num_bodies, 3)

        # Check if we have enough frames
        if self.max_motion_len < order + 1:
            # Not enough frames for specified order differences, return all zeros
            jitter = torch.zeros(
                self.num_motions, self.max_motion_len, device=self.device
            )
            return jitter

        if order == 1:
            # 1st order finite difference: data[t+1] - data[t]
            finite_diffs = (
                data_reshaped[:, 1:, :, :] - data_reshaped[:, :-1, :, :]
            )  # [num_motions, max_motion_len-1, num_bodies, 3]
            # Pad with zeros at the beginning: [num_motions, max_motion_len, num_bodies, 3]
            finite_diffs = torch.cat(
                [
                    torch.zeros(self.num_motions, 1, num_bodies, 3, device=self.device),
                    finite_diffs,
                ],
                dim=1,
            )
        elif order == 2:
            # 2nd order finite difference: data[t+1] - 2*data[t] + data[t-1]
            data_t_minus_1 = data_reshaped[
                :, :-2, :, :
            ]  # [num_motions, max_motion_len-2, num_bodies, 3]
            data_t = data_reshaped[
                :, 1:-1, :, :
            ]  # [num_motions, max_motion_len-2, num_bodies, 3]
            data_t_plus_1 = data_reshaped[
                :, 2:, :, :
            ]  # [num_motions, max_motion_len-2, num_bodies, 3]
            finite_diffs = (
                data_t_plus_1 - 2 * data_t + data_t_minus_1
            )  # [num_motions, max_motion_len-2, num_bodies, 3]
            # Pad with zeros at the beginning: [num_motions, max_motion_len, num_bodies, 3]
            finite_diffs = torch.cat(
                [
                    torch.zeros(self.num_motions, 2, num_bodies, 3, device=self.device),
                    finite_diffs,
                ],
                dim=1,
            )

        # Compute L2 norm for each body: [num_motions, max_motion_len, num_bodies]
        jitter_per_body = torch.norm(finite_diffs, dim=-1)

        # Aggregate across bodies
        if aggregate_method == "mean":
            jitter = jitter_per_body.mean(dim=-1)  # [num_motions, max_motion_len]
        elif aggregate_method == "max":
            jitter = jitter_per_body.max(dim=-1).values  # [num_motions, max_motion_len]
        elif aggregate_method == "sum":
            jitter = jitter_per_body.sum(dim=-1)  # [num_motions, max_motion_len]
        else:
            raise ValueError(f"Unknown aggregate_method: {aggregate_method}")

        # Apply the original mask to ensure jitter is 0 for invalid frames
        # Create a simple mask for valid frames
        frame_counts = self.frame_counts.unsqueeze(1)  # [num_motions, 1]
        indices = torch.arange(self.max_motion_len, device=self.device).unsqueeze(
            0
        )  # [1, max_motion_len]
        valid_frame_mask = indices < frame_counts  # [num_motions, max_motion_len]

        # Set jitter to 0 for invalid frames
        jitter = jitter.masked_fill(~valid_frame_mask, 0.0)

        return jitter

    def compute_jitter_reduce_each_motion(
        self, num_bodies: int, aggregate_method: str = "mean"
    ) -> torch.Tensor:
        """
        Compute jitter (2nd order finite differences of positions) and reduce across body dimensions.

        This method is specifically designed for rigid_body_pos data with shape [num_motions, max_motion_len, num_bodies*3].
        It computes the L2 norm of 2nd order finite differences (pos[t+1] - 2*pos[t] + pos[t-1]) for each body,
        then aggregates across all bodies using the specified method.
        Output is zero-padded at the beginning to match input length.

        Args:
            num_bodies: Number of rigid bodies (to reshape the flattened data)
            aggregate_method: How to aggregate across bodies ("mean", "max", "sum")

        Returns:
            torch.Tensor: Jitter values with shape [num_motions, max_motion_len] (same as input)
        """
        return self.compute_finite_difference_jitter_reduce_each_motion(
            num_bodies=num_bodies,
            aggregate_method=aggregate_method,
            order=2,
            field_description="rigid_body_pos",
        )

    def compute_rotation_jitter_reduce_each_motion(
        self, num_bodies: int, aggregate_method: str = "mean"
    ) -> torch.Tensor:
        """
        Compute rotation jitter (1st order finite differences of angular velocities) and reduce across body dimensions.

        This method is specifically designed for rigid_body_ang_vel data with shape [num_motions, max_motion_len, num_bodies*3].
        It computes the L2 norm of 1st order finite differences (ang_vel[t+1] - ang_vel[t]) for each body,
        then aggregates across all bodies using the specified method.
        Output is zero-padded at the beginning to match input length.

        Args:
            num_bodies: Number of rigid bodies (to reshape the flattened data)
            aggregate_method: How to aggregate across bodies ("mean", "max", "sum")

        Returns:
            torch.Tensor: Rotation jitter values with shape [num_motions, max_motion_len] (same as input)
        """
        return self.compute_finite_difference_jitter_reduce_each_motion(
            num_bodies=num_bodies,
            aggregate_method=aggregate_method,
            order=1,
            field_description="rigid_body_ang_vel",
        )

    def jitter_mean_reduce_each_motion(
        self, num_bodies: int, aggregate_method: str = "mean"
    ) -> torch.Tensor:
        """
        Compute jitter and then take the mean over time for each motion.

        Args:
            num_bodies: Number of rigid bodies
            aggregate_method: How to aggregate across bodies ("mean", "max", "sum")

        Returns:
            torch.Tensor: Mean jitter value for each motion [num_motions]
        """
        return self._generic_jitter_mean_reduce_each_motion(
            num_bodies=num_bodies,
            aggregate_method=aggregate_method,
            jitter_method=self.compute_jitter_reduce_each_motion,
        )

    def rotation_jitter_mean_reduce_each_motion(
        self, num_bodies: int, aggregate_method: str = "mean"
    ) -> torch.Tensor:
        """
        Compute rotation jitter and then take the mean over time for each motion.

        Args:
            num_bodies: Number of rigid bodies
            aggregate_method: How to aggregate across bodies ("mean", "max", "sum")

        Returns:
            torch.Tensor: Mean rotation jitter value for each motion [num_motions]
        """
        return self._generic_jitter_mean_reduce_each_motion(
            num_bodies=num_bodies,
            aggregate_method=aggregate_method,
            jitter_method=self.compute_rotation_jitter_reduce_each_motion,
        )

    def _generic_jitter_mean_reduce_each_motion(
        self, num_bodies: int, aggregate_method: str, jitter_method: Callable
    ) -> torch.Tensor:
        """
        Generic helper method to compute jitter and take the mean over time for each motion.

        Args:
            num_bodies: Number of rigid bodies
            aggregate_method: How to aggregate across bodies ("mean", "max", "sum")
            jitter_method: The method to call for computing jitter

        Returns:
            torch.Tensor: Mean jitter value for each motion [num_motions]
        """
        jitter = jitter_method(
            num_bodies, aggregate_method
        )  # [num_motions, max_motion_len]

        # Apply the unfilled mask to only consider valid frames
        mask = self.get_unfilled_mask()[
            :, :, 0
        ]  # [num_motions, max_motion_len] (use first sub-feature mask)
        jitter_masked = jitter.masked_fill(mask, 0.0)

        # Sum jitter values and divide by valid count
        jitter_sum = jitter_masked.sum(dim=1)  # [num_motions]
        valid_frame_counts = self.frame_counts.clamp(
            min=1
        )  # [num_motions], clamp to avoid division by zero

        mean_jitter = jitter_sum / valid_frame_counts  # [num_motions]

        # Set mean to 0 for motions with no valid frames
        zero_frame_mask = self.frame_counts == 0
        mean_jitter = mean_jitter.masked_fill(zero_frame_mask, 0.0)

        return mean_jitter

    def copy_from(
        self,
        other: "MotionMetrics",
    ) -> None:
        """Copy data from another MotionMetrics object."""
        self.data = other.data.clone()
        self.frame_counts = other.frame_counts.clone()

    def copy_from_motion_ids(
        self,
        other: "MotionMetrics",
        motion_ids: torch.Tensor,
    ) -> None:
        """Copy data from another MotionMetrics object for specific motions."""
        self.data[motion_ids] = other.data[motion_ids]
        self.frame_counts[motion_ids] = other.frame_counts[motion_ids]

    def merge_from(
        self,
        other: "MotionMetrics",
    ) -> None:
        """Merge data from another MotionMetrics object."""
        assert self.max_motion_len == other.max_motion_len
        assert self.num_sub_features == other.num_sub_features

        self.data = torch.cat([self.data, other.data], dim=0)
        self.frame_counts = torch.cat([self.frame_counts, other.frame_counts], dim=0)
        self.motion_lens = torch.cat([self.motion_lens, other.motion_lens], dim=0)
        self.num_motions = self.data.shape[0]

    def reset(self) -> None:
        """Reset all stored data and frame counts."""
        self.data.zero_()
        self.frame_counts.zero_()

    def to(self, device: torch.device) -> "MotionMetrics":
        """Move metrics to specified device."""
        self.device = device
        self.data = self.data.to(device)
        self.frame_counts = self.frame_counts.to(device)
        return self
