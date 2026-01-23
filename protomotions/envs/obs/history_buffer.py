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
"""History buffer for temporal observations.

This module provides the HistoryBuffer class used by the generic observation
history system in BaseEnv.
"""

import torch
from torch import Tensor, nn


class HistoryBuffer(nn.Module):
    """Circular buffer for storing temporal history of observations or actions.

    Stores the past N frames of data where index 0 is the most recent frame.
    Used for temporal observations like action history or historical poses.

    Args:
        num_steps: Number of historical timesteps to store.
        num_envs: Number of parallel environments.
        shape: Shape of each data element (default: scalar).
        dtype: Data type for storage.
        device: Device for tensor storage.
    """

    data: Tensor

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        shape: tuple = (),
        dtype=torch.float,
        device="cpu",
    ):
        super().__init__()
        data = torch.zeros(num_steps, num_envs, *shape, dtype=dtype, device=device)
        self.register_buffer("data", data, persistent=False)
        self.to(device)

    def rotate(self):
        """Shift history forward by one timestep (oldest frame is discarded)."""
        self.data = self.data.roll(
            shifts=1, dims=0
        )  # equivalent to self.data[i + 1] = self.data[i]

    @torch.no_grad()
    def update(self, fresh_data: Tensor):
        """Rotate buffer and update current frame with new data.

        Args:
            fresh_data: New data to insert at current frame [num_envs, *shape]
        """
        self.rotate()
        self.set_curr(fresh_data)

    @torch.no_grad()
    def set_all(self, fresh_data: Tensor, env_ids=slice(None)):
        """Set all timesteps for specified environments.

        Args:
            fresh_data: Data for all timesteps [num_steps, num_envs, *shape]
            env_ids: Environment indices to update
        """
        self.data[:, env_ids] = fresh_data

    @torch.no_grad()
    def set_hist(self, fresh_data: Tensor, env_ids=slice(None)):
        """Set historical data (excluding current frame).

        Args:
            fresh_data: Historical data [num_steps-1, num_envs, *shape]
            env_ids: Environment indices to update
        """
        self.data[1:, env_ids] = fresh_data

    @torch.no_grad()
    def set_curr(self, fresh_data: Tensor, env_ids=slice(None)):
        """Set current frame data.

        Args:
            fresh_data: Current frame data [num_envs, *shape]
            env_ids: Environment indices to update
        """
        self.data[0, env_ids] = fresh_data

    def get_hist(self, env_ids=slice(None)):
        """Get historical data (excluding current frame).

        Args:
            env_ids: Environment indices to retrieve

        Returns:
            Historical data [num_steps-1, num_envs, *shape]
        """
        return self.data[1:, env_ids]

    def get_current(self, env_ids=slice(None)):
        """Get current frame data.

        Args:
            env_ids: Environment indices to retrieve

        Returns:
            Current frame data [num_envs, *shape]
        """
        return self.data[0, env_ids]

    def get_all(self, env_ids=slice(None)):
        """Get all timesteps.

        Args:
            env_ids: Environment indices to retrieve

        Returns:
            All historical data [num_steps, num_envs, *shape]
        """
        return self.data[:, env_ids]

    def get_all_flattened(self, env_ids=slice(None)):
        """Get all timesteps flattened into a single feature vector per environment.

        Args:
            env_ids: Environment indices to retrieve

        Returns:
            Flattened history [num_envs, num_steps * features]
        """
        data = self.get_all(env_ids)
        num_envs = data.shape[1]
        return data.permute(1, 0, 2).reshape(num_envs, -1)

    def get_index(self, idx: int, env_ids=slice(None)):
        """Get data at specific timestep index.

        Args:
            idx: Timestep index (0 = current, 1 = previous, etc.)
            env_ids: Environment indices to retrieve

        Returns:
            Data at specified timestep [num_envs, *shape]
        """
        return self.data[idx, env_ids]

    @property
    def device(self) -> torch.device:
        """Get device from registered buffers."""
        return self.data.device
