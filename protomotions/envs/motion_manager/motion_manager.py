# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
#    - Their weights are 0 from construction on; `set_sampling_weights`, the
#      one writer of the weight vector, keeps them at 0 whatever it is given
#    - Can be used during training or evaluation
#    - Works with any subset_method configuration


import torch
import numpy as np
from omegaconf.listconfig import ListConfig
from protomotions.components.motion_lib import MotionLib
from protomotions.envs.motion_manager.config import MotionManagerConfig
from typing import Optional, Tuple


class MotionManager:
    """Manages motion sampling and tracking for environments.

    Handles sampling of reference motions, time progression, and weighted sampling based on performance.
    Supports motion subsetting and exclusion for focused training/evaluation.
    ``set_sampling_weights`` is the only writer of ``motion_weights`` (the
    curriculum, a resumed checkpoint and the library's initial weights all go
    through it), over an eligibility mask fixed at construction.

    ``motion_ids`` and ``motion_times`` hold the per-env clip assignment. To
    pin clips from outside, use ``get_assignment`` / ``set_assignment`` /
    ``assign``; they update the tensors in place so everyone holding them
    sees the change.

    Args:
        config: Configuration object for motion manager.
        num_envs: Number of parallel environments.
        env_dt: Environment timestep.
        device: Device for tensor storage.
        motion_lib: Motion library containing reference motions.
        fixed_motion_ids_per_env: Optional fixed motion IDs per environment (-1 for random sampling).
    """

    def __init__(
        self,
        config: MotionManagerConfig,
        num_envs: int,
        env_dt: float,
        device: torch.device,
        motion_lib: MotionLib,
        fixed_motion_ids_per_env: Optional[torch.Tensor] = None,
        *,
        generator: Optional[torch.Generator],
    ):
        self.config = config
        self.num_envs = num_envs
        self.device = device
        self.motion_lib = motion_lib
        self.env_dt = env_dt
        # The motion-sampling stage's generator (None: the global stream);
        # every draw below passes it.
        self.generator = generator

        # Who writes motion_ids / motion_times: other objects keep references
        # to the two tensors, so nothing rebinds them. Resets choose clips
        # through sample_motions; the evaluator pins clips through
        # get_assignment / set_assignment / assign; the kinematic replay
        # control component (kinematic_replay_control.py) zeroes motion_times
        # on a user reset and advances it every step. All write in place.
        self.motion_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.motion_times = torch.zeros(num_envs, device=device)

        # Sampling vectors
        self.init_start_probs = (
            torch.ones(num_envs, dtype=torch.float, device=device)
            * self.config.init_start_prob
        )

        # Handle motion subsetting (only affects sampling, not weights)
        self._setup_motion_subset()

        # Which clips may ever be sampled, fixed at construction: not excluded
        # by the config, and long enough for the reset window when
        # sample_time_truncate_s is set (reset sampling truncates the window by
        # truncate_s + env_dt; a shorter clip has no valid window and would be
        # forced to t=0, see sample_time's clamp). ``set_sampling_weights`` is
        # the one writer of the weight vector and keeps every other clip at 0.
        self._setup_motion_exclusion()
        eligible = torch.ones(
            self.motion_lib.num_motions(), dtype=torch.bool, device=device
        )
        if self.excluded_motion_ids is not None:
            eligible[self.excluded_motion_ids] = False
        if self.config.sample_time_truncate_s is not None:
            min_length = float(self.config.sample_time_truncate_s) + self.env_dt
            eligible &= self.motion_lib.motion_lengths.to(device=device) >= min_length
        self._eligible = eligible

        # The initial weights are the library's (a user may specify them there),
        # through the writer so the ineligible clips start at 0.
        initial_weights = self.motion_lib.motion_weights.to(device=device)
        self.motion_weights = torch.zeros_like(initial_weights)
        self.set_sampling_weights(initial_weights, floor=0.0, ceiling=float("inf"))

        # Handle fixed motion IDs for scene-motion correspondence
        self._setup_fixed_motion_ids(fixed_motion_ids_per_env)

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

    def _setup_fixed_motion_ids(self, fixed_motion_ids_per_env: Optional[torch.Tensor]):
        """Setup fixed motion IDs. Use -1 for environments that should use random sampling."""
        self._fixed_motion_ids_per_env = fixed_motion_ids_per_env
        self._env_has_fixed_motion = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        if fixed_motion_ids_per_env is not None:
            assert fixed_motion_ids_per_env.shape == (
                self.num_envs,
            ), f"fixed_motion_ids_per_env must be of shape ({self.num_envs},), got {fixed_motion_ids_per_env.shape}"
            num_motions = self.motion_lib.num_motions()
            ids = fixed_motion_ids_per_env
            bad = (ids != -1) & ((ids < 0) | (ids >= num_motions))
            if bool(bad.any()):
                env = int(bad.nonzero()[0])
                raise ValueError(
                    f"environment {env} is paired to clip {int(ids[env])}, but "
                    f"the loaded motion library has {num_motions} clips"
                )
            self._env_has_fixed_motion = fixed_motion_ids_per_env != -1
            num_fixed = self._env_has_fixed_motion.sum().item()
            if num_fixed > 0:
                print(f"Motion Manager: {num_fixed} environments have fixed motion IDs")

    @property
    def fixed_motion_ids_per_env(self) -> Optional[torch.Tensor]:
        """Per-env pinned clip (-1 where the env samples), or None when nothing is pinned."""
        return self._fixed_motion_ids_per_env

    def get_assignment(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Clones of ``(motion_ids, motion_times)``: a snapshot later writes cannot change.

        This is the supported way for outside code (the evaluator) to save the
        clip assignment before overwriting it; ``set_assignment`` puts it back.
        """
        return self.motion_ids.clone(), self.motion_times.clone()

    def set_assignment(
        self, motion_ids: torch.Tensor, motion_times: torch.Tensor
    ) -> None:
        """Overwrite every env's clip and time in place.

        Both tensors must have shape ``(num_envs,)``. The values are copied into
        the existing ``motion_ids`` / ``motion_times`` tensors, never rebound,
        because other objects hold references to them. This is the supported
        way for outside code (the evaluator) to pin clips; ``sample_motions``
        stays the way resets choose clips.
        """
        expected = (self.num_envs,)
        for name, value in (("motion_ids", motion_ids), ("motion_times", motion_times)):
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {tuple(value.shape)}"
                )
        self.motion_ids.copy_(motion_ids)
        self.motion_times.copy_(motion_times)

    def assign(
        self,
        env_ids: torch.Tensor,
        *,
        motion_ids: Optional[torch.Tensor] = None,
        motion_times: Optional[torch.Tensor] = None,
    ) -> None:
        """Write the clip and/or time of ``env_ids`` only, in place.

        Each given tensor must match ``env_ids.shape``; at least one must be
        given. Rows outside ``env_ids`` are untouched. Like ``set_assignment``,
        this is for outside code that pins clips; resets use ``sample_motions``.
        """
        if motion_ids is None and motion_times is None:
            raise ValueError("assign needs motion_ids, motion_times or both")
        for name, value in (("motion_ids", motion_ids), ("motion_times", motion_times)):
            if value is not None and value.shape != env_ids.shape:
                raise ValueError(
                    f"{name} must match env_ids.shape {tuple(env_ids.shape)}, "
                    f"got {tuple(value.shape)}"
                )
        if motion_ids is not None:
            self.motion_ids[env_ids] = motion_ids
        if motion_times is not None:
            self.motion_times[env_ids] = motion_times

    def get_unique_fixed_motions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get unique fixed motion IDs and the first environment index for each.

        Returns:
            Tuple of (unique_motion_ids, first_env_indices). Excludes -1 entries.
        """
        empty_result = (
            torch.tensor([], device=self.device, dtype=torch.long),
            torch.tensor([], device=self.device, dtype=torch.long),
        )

        if self._fixed_motion_ids_per_env is None:
            return empty_result

        valid_mask_torch = self._fixed_motion_ids_per_env != -1

        if not valid_mask_torch.any():
            return empty_result

        fixed_motion_ids_np = self._fixed_motion_ids_per_env.cpu().numpy()
        valid_mask_np = valid_mask_torch.cpu().numpy()

        valid_motion_ids_np = fixed_motion_ids_np[valid_mask_np]
        all_env_indices_np = np.arange(len(self._fixed_motion_ids_per_env))
        valid_env_indices_np = all_env_indices_np[valid_mask_np]

        unique_motion_ids_np, first_indices_in_valid_np = np.unique(
            valid_motion_ids_np, return_index=True
        )

        first_env_indices_np = valid_env_indices_np[first_indices_in_valid_np]

        first_env_indices_torch = torch.from_numpy(first_env_indices_np).to(
            device=self.device, dtype=torch.long
        )
        unique_motion_ids_torch = torch.from_numpy(unique_motion_ids_np).to(
            device=self.device, dtype=torch.long
        )

        return unique_motion_ids_torch, first_env_indices_torch

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
                    print(
                        f"Warning: No rank-0 failed motions file found in {failed_motions_dir}"
                    )
                    return None
            else:
                print(
                    f"Warning: No rank-0 failed motions file found in {failed_motions_dir}"
                )
                return None
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

    def set_sampling_weights(
        self, weights: torch.Tensor, *, floor: float, ceiling: float
    ) -> None:
        """The one writer of ``motion_weights``.

        Every eligible clip (not excluded, not too short for the reset window)
        gets ``weights`` clamped to ``[floor, ceiling]``; every other clip
        stays at 0, whatever ``weights`` says. The evaluator's curriculum, a
        resumed checkpoint and the library's initial weights all pass through
        here, so the eligibility mask can never be undone by a write.
        """
        num_motions = self.motion_weights.shape[0]
        if tuple(weights.shape) != (num_motions,):
            raise ValueError(
                f"sampling weights must have shape ({num_motions},), got "
                f"{tuple(weights.shape)}"
            )
        if not 0 <= floor <= ceiling:
            raise ValueError("floor must satisfy 0 <= floor <= ceiling")
        clamped = weights.to(
            device=self.motion_weights.device, dtype=self.motion_weights.dtype
        ).clamp(min=floor, max=ceiling)
        self.motion_weights[:] = torch.where(
            self._eligible, clamped, torch.zeros_like(clamped)
        )

    def sample_n_motion_ids(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample motion IDs from weighted distribution.

        Args:
            n: Number of motion IDs to sample

        Returns:
            Sampled motion IDs [n]
        """
        if n == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        # The weight vector already carries 0 for every ineligible clip.
        return torch.multinomial(
            self.motion_weights,
            num_samples=n,
            replacement=True,
            generator=self.generator,
        )

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
        phase = torch.rand(
            motion_ids.shape, device=self.device, generator=self.generator
        )

        max_time = self.motion_lib.motion_lengths[motion_ids].clone()

        if truncate_time is not None:
            assert torch.all(torch.tensor(truncate_time) >= 0.0)
            max_time -= truncate_time
            max_time = max_time.clamp(min=0.0)  # clips shorter than truncate_time start at t=0

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
            # Apply fixed motion IDs if not overridden
            if new_motion_ids is None and self._fixed_motion_ids_per_env is not None:
                if any(self._env_has_fixed_motion[env_ids]):
                    new_motion_ids = self._fixed_motion_ids_per_env[env_ids]

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

        # Truncate reset sampling by env_dt plus the optional configured margin so
        # random reset times stay away from the end of the clip (matched by the
        # too-short-clip weight zeroing in __init__).
        reset_truncate = self.env_dt
        if self.config.sample_time_truncate_s is not None:
            reset_truncate += float(self.config.sample_time_truncate_s)
        new_times = self.sample_time(new_motion_ids, truncate_time=reset_truncate)

        if self.config.init_start_prob > 0:
            init_start = torch.bernoulli(
                self.init_start_probs[: len(env_ids)], generator=self.generator
            )
            new_times = torch.where(
                init_start == 1,
                torch.zeros_like(new_times),
                new_times,
            )

        self.motion_ids[env_ids] = new_motion_ids
        self.motion_times[env_ids] = new_times

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
                self.set_sampling_weights(
                    state_dict["motion_weights"], floor=0.0, ceiling=float("inf")
                )
