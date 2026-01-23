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
# Motion Manager Configuration

# ## Subset Method

# The motion manager now supports filtering specific motion IDs during evaluation using the `subset_method` parameter.

# ### Configuration Options

# 1. **`subset_method: null`** (default)

# 2. **`subset_method: "first"`**
#    - Uses the first N motions where N = num_envs

# 3. **`subset_method: "last"`**
#    - Uses the last N motions where N = num_envs

# 4. **`subset_method: "random"`**
#    - Randomly selects N motions where N = num_envs
#    - The selection is fixed per session (not re-randomized)

# 5. **`subset_method: [0, 1, 5, 10]`**
#    - Uses specific motion IDs as provided in the list
#    - List length must equal num_envs for deterministic assignment
#    - Each environment gets a specific motion (no random sampling)

# ## Exclude Motion IDs

# The motion manager supports excluding specific motion IDs from probabilistic sampling using the `exclude_motion_ids` parameter.

# ### Configuration Options

# 1. **`exclude_motion_ids: null`** (default)
#    - No motions are excluded from sampling

# 2. **`exclude_motion_ids: [2, 5, 8, 12]`**
#    - Excludes the specified motion IDs from probabilistic sampling
#    - Sets their weights to 0 before each sampling operation
#    - Persistent exclusion even if motion weights are updated by other classes
#    - Can be used during training or evaluation
#    - Works with any subset_method configuration


import torch
from omegaconf.listconfig import ListConfig
from protomotions.components.motion_lib import MotionLib
from protomotions.envs.motion_manager.config import MotionManagerConfig
from typing import Optional, Tuple


class MotionManager:
    """Manages motion sampling and tracking for environments.

    Handles sampling of reference motions, time progression, and weighted sampling based on performance.
    Supports motion subsetting and exclusion for focused training/evaluation.

    Args:
        config: Configuration object for motion manager.
        num_envs: Number of parallel environments.
        env_dt: Environment timestep.
        device: Device for tensor storage.
        motion_lib: Motion library containing reference motions.
    """

    def __init__(
        self,
        config: MotionManagerConfig,
        num_envs: int,
        env_dt: float,
        device: torch.device,
        motion_lib: MotionLib,
    ):
        self.config = config
        self.num_envs = num_envs
        self.device = device
        self.motion_lib = motion_lib
        self.env_dt = env_dt

        self.motion_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.motion_times = torch.zeros(num_envs, device=device)

        # Sampling vectors
        self.init_start_probs = (
            torch.ones(num_envs, dtype=torch.float, device=device)
            * self.config.init_start_prob
        )

        # initialize motion weights from motion lib, in case user specified initial weights there
        self.motion_weights = self.motion_lib.motion_weights.clone().to(device=device)

        # Handle motion subsetting (only affects sampling, not weights)
        self._setup_motion_subset()

        # Handle motion exclusion (store excluded IDs to apply during sampling)
        self._setup_motion_exclusion()

    def _setup_motion_subset(self):
        """
        Setup motion subset based on config.subset_method
        Only affects which motion IDs are sampled, does not modify motion weights.
        """
        subset_method = self.config.subset_method

        total_motions = self.motion_lib.num_motions()
        if subset_method is None or self.num_envs > total_motions:
            # Use all motions
            self.available_motion_ids = None
            return

        if isinstance(subset_method, str):
            if subset_method == "first":
                # Use first min(num_envs, total_motions) motions
                n_motions = min(self.num_envs, total_motions)
                self.available_motion_ids = torch.arange(n_motions, device=self.device)
            elif subset_method == "last":
                # Use last min(num_envs, total_motions) motions
                n_motions = min(self.num_envs, total_motions)
                self.available_motion_ids = torch.arange(
                    total_motions - n_motions, total_motions, device=self.device
                )
            elif subset_method == "random":
                # Randomly sample min(num_envs, total_motions) motions
                n_motions = min(self.num_envs, total_motions)
                self.available_motion_ids = torch.randperm(
                    total_motions, device=self.device
                )[:n_motions]
            else:
                raise ValueError(
                    f"Unknown subset_method string: {subset_method}. Must be 'first', 'last', or 'random'"
                )
        elif isinstance(subset_method, (list, ListConfig)):
            # Use specific motion IDs
            motion_ids = list(subset_method)
            # Assert that the number of motion IDs matches num_envs
            if len(motion_ids) != self.num_envs:
                raise ValueError(
                    f"When using list for subset_method, length ({len(motion_ids)}) must equal num_envs ({self.num_envs})"
                )
            # Validate motion IDs
            for motion_id in motion_ids:
                if motion_id < 0 or motion_id >= total_motions:
                    raise ValueError(
                        f"Motion ID {motion_id} is out of range [0, {total_motions-1}]"
                    )
            self.available_motion_ids = torch.tensor(
                motion_ids, dtype=torch.long, device=self.device
            )
        else:
            raise ValueError(
                f"subset_method must be None, string, or list of integers, got {type(subset_method)}"
            )

        if self.available_motion_ids is not None:
            print(
                f"Motion Manager: Using subset of {len(self.available_motion_ids)} motions out of {total_motions} total motions"
            )
            print(f"Available motion IDs: {self.available_motion_ids.cpu().tolist()}")

    def _setup_motion_exclusion(self):
        """
        Setup motion exclusion based on config.exclude_motions_file and/or config.exclude_motion_ids.
        If both are provided, combines them (union).
        """
        excluded_set = set()

        if self.config.exclude_motions_file is not None:
            file_ids = self._load_exclusions_from_file(self.config.exclude_motions_file)
            if file_ids:
                excluded_set.update(file_ids)

        if self.config.exclude_motion_ids is not None:
            if isinstance(self.config.exclude_motion_ids, (list, ListConfig)):
                excluded_set.update(self.config.exclude_motion_ids)
            else:
                raise ValueError(
                    f"exclude_motion_ids must be None or list of integers, got {type(self.config.exclude_motion_ids)}"
                )

        if len(excluded_set) == 0:
            self.excluded_motion_ids = None
            return

        motion_ids = sorted(excluded_set)
        total_motions = self.motion_lib.num_motions()
        for motion_id in motion_ids:
            if motion_id < 0 or motion_id >= total_motions:
                raise ValueError(
                    f"Exclude motion ID {motion_id} is out of range [0, {total_motions-1}]"
                )

        self.excluded_motion_ids = torch.tensor(
            motion_ids, dtype=torch.long, device=self.device
        )

        print(f"Motion Manager: Excluding {len(motion_ids)} motions from sampling")
        if len(motion_ids) <= 50:
            print(f"Excluded motion IDs: {motion_ids}")

    def _load_exclusions_from_file(self, file_path: str) -> Optional[list]:
        """Load motion IDs to exclude from a file (one ID per line)."""
        import re
        from pathlib import Path

        path = Path(file_path)

        if path.is_file():
            # Direct file path provided
            pass
        elif path.is_dir():
            # Directory provided - look for failed_motions subdirectory
            failed_motions_dir = path / "failed_motions"
            if failed_motions_dir.exists():
                pattern = re.compile(r"failed_motions_epoch_(\d+)_rank_0\.txt")
                epoch_files = []
                for fp in failed_motions_dir.iterdir():
                    match = pattern.match(fp.name)
                    if match:
                        epoch_files.append((int(match.group(1)), fp))
                if epoch_files:
                    epoch_files.sort(key=lambda x: x[0], reverse=True)
                    path = epoch_files[0][1]
                    print(f"Motion Manager: Using failed motions from {path}")
        else:
            print(f"Warning: exclude_motions_file not found: {path}")
            return None

        if not path.exists():
            print(f"Warning: exclude_motions_file not found: {path}")
            return None

        with open(path, "r") as f:
            motion_ids = [int(line.strip()) for line in f if line.strip()]
        print(f"Motion Manager: Loaded {len(motion_ids)} motion IDs to exclude from {path}")
        return motion_ids

    def _apply_motion_exclusions(self):
        """
        Apply motion exclusions by setting weights of excluded motions to 0.
        This should be called before each sampling operation.
        """
        if self.excluded_motion_ids is not None:
            self.motion_weights[self.excluded_motion_ids] = 0.0

    def sample_n_motion_ids(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample motion IDs from weighted distribution.

        Args:
            n: Number of motion IDs to sample

        Returns:
            Sampled motion IDs [n]
        """
        # Apply exclusions before sampling
        self._apply_motion_exclusions()
        return torch.multinomial(self.motion_weights, num_samples=n, replacement=True)

    def sample_time(
        self, motion_ids: torch.Tensor, truncate_time: Optional[float] = None
    ) -> torch.Tensor:
        """Sample random start times for motions.

        Args:
            motion_ids: Motion IDs to sample times for [num_samples]
            truncate_time: If specified, truncate max time by this amount

        Returns:
            Sampled times [num_samples]
        """
        phase = torch.rand(motion_ids.shape, device=self.device)

        max_time = self.motion_lib.motion_lengths[motion_ids].clone()

        if truncate_time is not None:
            assert torch.all(torch.tensor(truncate_time) >= 0.0)
            max_time -= truncate_time
            assert torch.all(max_time >= 0)

        motion_time = phase * max_time

        return motion_time

    def sample_motions(
        self, env_ids: torch.Tensor, new_motion_ids: Optional[torch.Tensor] = None
    ):
        """
        Reset the motion for a set of environments.
        This method handles the process of resetting the motion for a specified set of environments.
        It ensures that the reset process is correctly handled based on the current configuration.

        Args:
            env_ids (Tensor): Indices of the environments to reset.
            new_motion_ids (Tensor, optional):
                Force new motion IDs for the reset environments.
                **In the same shape as env_ids**
                -1 indicates random sampling
        Returns:
            None
        """

        # Handle subset case - deterministic assignment
        if self.available_motion_ids is not None:
            assert (
                len(self.available_motion_ids) == self.num_envs
            ), f"Available motion IDs length ({len(self.available_motion_ids)}) must equal num_envs ({self.num_envs})"
            new_motion_ids = self.available_motion_ids[env_ids].to(self.device)
        else:
            # Handle normal case - random sampling with optional override
            if new_motion_ids is not None:
                assert (
                    new_motion_ids.shape == env_ids.shape
                ), "new_motion_ids must be the same shape as env_ids"
                # Find environments that need random sampling (-1 indicates random sampling)
                need_random = new_motion_ids == -1
                num_random = need_random.sum()

                if num_random > 0:
                    random_motion_ids = self.sample_n_motion_ids(num_random)
                    new_motion_ids = new_motion_ids.clone().to(self.device)
                    new_motion_ids[need_random] = random_motion_ids
                else:
                    new_motion_ids = new_motion_ids.to(self.device)
            else:
                # Pure random sampling
                new_motion_ids = self.sample_n_motion_ids(len(env_ids))

        new_times = self.sample_time(new_motion_ids, truncate_time=self.env_dt)

        if self.config.init_start_prob > 0:
            init_start = torch.bernoulli(self.init_start_probs[: len(env_ids)])
            new_times = torch.where(
                init_start == 1,
                torch.zeros_like(new_times),
                new_times,
            )

        self.motion_ids[env_ids] = new_motion_ids
        self.motion_times[env_ids] = new_times

    def update_sampling_weights(self, weights: torch.Tensor):
        k = min(50, weights.shape[0])
        topk = torch.topk(weights, k, largest=True)
        print(f"Top {k} weights:")
        print("Indices:", topk.indices)
        print("Values:", topk.values)

        self.motion_weights[:] = weights
        # Apply exclusions after updating weights
        self._apply_motion_exclusions()

    def get_state_dict(self):
        state_dict = {
            "motion_file_name": self.motion_lib.motion_file,
            "motion_weights": self.motion_weights.cpu().clone(),
        }
        return state_dict

    def load_state_dict(self, state_dict):
        if "motion_weights" in state_dict and "motion_file_name" in state_dict:
            if state_dict["motion_file_name"] != self.motion_lib.motion_file:
                # should match given we have task id, but double check here
                print(
                    f"Warning: skip loading motion weights due to motion file name mismatch: {state_dict['motion_file_name']} != {self.motion_lib.motion_file}"
                )
            else:
                self.motion_weights[:] = state_dict["motion_weights"].to(
                    self.motion_weights.device
                )
