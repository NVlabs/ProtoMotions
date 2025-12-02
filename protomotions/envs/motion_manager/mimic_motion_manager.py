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
from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
from protomotions.envs.motion_manager.motion_manager import MotionManager
from protomotions.components.motion_lib import MotionLib
import torch
from typing import Optional, Tuple
import numpy as np


class MimicMotionManager(MotionManager):
    """Motion manager specialized for mimic environments with fixed motion support.

    Extends the base MotionManager to handle mimic-specific motion sampling,
    including support for fixed motion assignments per environment and
    conditional resampling on reset.
    """

    config: MimicMotionManagerConfig

    def __init__(
        self,
        config: MimicMotionManagerConfig,
        num_envs: int,
        env_dt: float,
        device: torch.device,
        motion_lib: MotionLib,
        fixed_motion_ids_per_env: Optional[torch.Tensor] = None,
    ):
        """A motion manager that handles motion sampling and tracking for mimic environments.

        Args:
            config: Configuration object containing motion manager settings
            num_envs (int): Number of parallel environments
            env_dt (float): Environment timestep
            device (torch.device): Device to store tensors on
            motion_lib (MotionLib): Motion library containing reference motions
            fixed_motion_ids_per_env (Optional[torch.Tensor], optional): If provided, specifies fixed motion IDs to use for each environment. Defaults to None.
        """
        super().__init__(config, num_envs, env_dt, device, motion_lib)

        self._fixed_motion_ids_per_env = fixed_motion_ids_per_env
        self._env_has_fixed_motion = torch.zeros(
            num_envs, device=device, dtype=torch.bool
        )
        if fixed_motion_ids_per_env is not None:
            assert fixed_motion_ids_per_env.shape == (
                num_envs,
            ), "fixed_motion_ids_per_env must be of shape (num_envs,)"
            self._env_has_fixed_motion = fixed_motion_ids_per_env != -1

    def get_unique_fixed_motions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get unique fixed motion IDs and the first environment index for each, using NumPy.

        Returns:
            Tuple containing:
                - Tensor of unique motion IDs (excluding -1 which indicates no fixed motion)
                - Tensor of first environment indices for each unique motion ID
        """
        empty_result = (
            torch.tensor([], device=self.device, dtype=torch.long),
            torch.tensor([], device=self.device, dtype=torch.long),
        )

        if self._fixed_motion_ids_per_env is None:
            return empty_result

        # Mask for valid motion IDs (not -1)
        valid_mask_torch = self._fixed_motion_ids_per_env != -1

        if not valid_mask_torch.any():
            return empty_result

        # Convert to NumPy to use NumPy's unique function
        fixed_motion_ids_np = self._fixed_motion_ids_per_env.cpu().numpy()
        valid_mask_np = valid_mask_torch.cpu().numpy()

        # Get the valid motion IDs in NumPy
        valid_motion_ids_np = fixed_motion_ids_np[valid_mask_np]

        # Get the original environment indices corresponding to valid motions (keep as Torch tensor)
        all_env_indices_np = np.arange(len(self._fixed_motion_ids_per_env))
        valid_env_indices_np = all_env_indices_np[valid_mask_np]

        # Find unique motion IDs and the indices of their first occurrence within the *valid* NumPy array
        unique_motion_ids_np, first_indices_in_valid_np = np.unique(
            valid_motion_ids_np, return_index=True
        )

        # Get the corresponding environment indices using the first occurrence indices (NumPy indices -> Torch tensor)
        first_env_indices_np = valid_env_indices_np[first_indices_in_valid_np]

        first_env_indices_torch = torch.from_numpy(first_env_indices_np).to(
            device=self.device, dtype=torch.long
        )
        unique_motion_ids_torch = torch.from_numpy(unique_motion_ids_np).to(
            device=self.device, dtype=torch.long
        )

        return unique_motion_ids_torch, first_env_indices_torch

    def get_done_tracks(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Check which motion tracks have reached their end time.

        Args:
            env_ids: Optional tensor of environment indices to check. If None, checks all environments.

        Returns:
            Boolean tensor indicating which motion tracks are done (True) or still playing (False)
        """
        end_times = self.motion_lib.motion_lengths[self.motion_ids]
        done_clip = (self.motion_times + self.env_dt) >= end_times
        if env_ids is not None:
            done_clip = done_clip[env_ids]
        return done_clip

    def post_physics_step(self):
        """Advance motion playback time by one environment timestep.

        Called after each physics simulation step to update the current time
        in each motion track.
        """
        self.motion_times += self.env_dt

    def sample_motions(
        self, env_ids: torch.Tensor, new_motion_ids: Optional[torch.Tensor] = None
    ):
        """Sample new motions for environments.

        This overrides the base class to:
        1. Handle mimic-specific resample_on_reset logic
        2. Respect fixed motion IDs assigned to environments

        Args:
            env_ids (Tensor): Indices of the environments to reset.
            new_motion_ids (Tensor, optional):
                Force new motion IDs for the reset environments.
                If provided, this overrides the fixed motion IDs.
        """
        # Mimic-specific: Only resample motions that have finished if resample_on_reset is False
        reset_env_ids = env_ids
        if not self.config.resample_on_reset:
            done_tracks = self.get_done_tracks(env_ids)
            reset_env_ids = env_ids[done_tracks]

        # Only proceed if there are environments to reset
        if len(reset_env_ids) == 0:
            return

        # Handle fixed motion IDs for scene-based environments
        if new_motion_ids is None and self._env_has_fixed_motion is not None:
            # Check if any of the reset envs have fixed motions
            if any(self._env_has_fixed_motion[reset_env_ids]):
                # Use fixed motion IDs for those environments
                new_motion_ids = self._fixed_motion_ids_per_env[reset_env_ids]

        # Call parent sample_motions with the filtered env_ids
        super().sample_motions(reset_env_ids, new_motion_ids)
