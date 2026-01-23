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
"""Motion library for managing reference motion data.

This module provides the MotionLib class which stores and manages collections of
motion clips for use in motion tracking and imitation learning. It supports efficient
loading, sampling, and interpolation of motion data from various formats (.motion, .yaml, .pt).

Key Classes:
    - MotionLib: Main motion library class for motion management

Key Features:
    - Load motions from .motion files, YAML configs, or packaged .pt files
    - Weighted sampling of motions
    - Frame interpolation for smooth motion queries
    - Batched access for parallel environments
    - Distributed training support
"""

import os
from typing import Optional, Tuple
from easydict import EasyDict
from pathlib import Path

import torch
import yaml
from lightning_fabric.utilities.rank_zero import _get_rank

from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    StateConversion,
)
from protomotions.utils.rotations import quat_to_exp_map
from dataclasses import dataclass, field

from protomotions.utils.motion_interpolation_utils import (
    interpolate_pos,
    interpolate_quat,
    calc_frame_blend,
)

# Mapping from MotionLib (packaged motion) field names to RobotState (single motion/sim state) field names
_motion_field_mapping = {
    "gts": "rigid_body_pos",
    "grs": "rigid_body_rot",
    "gavs": "rigid_body_ang_vel",
    "gvs": "rigid_body_vel",
    "dvs": "dof_vel",
    "dps": "dof_pos",
    "contacts": "rigid_body_contacts",
}


@dataclass
class MotionLibConfig:
    """Configuration for motion library."""

    _target_: str = "protomotions.components.motion_lib.MotionLib"
    motion_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to motion file (.pt, .yaml, or .motion). None for empty library."}
    )
    world_size: int = field(
        default=1,
        metadata={"help": "World size for distributed training (sharded loading).", "min": 1}
    )
    get_motion_state_use_blend: bool = field(
        default=True,
        metadata={"help": "Use interpolation for smooth motion queries between frames."}
    )


class MotionLib:
    """Motion library for managing and sampling reference motion data.

    Stores and manages a collection of motion clips for use in imitation learning.
    Supports efficient sampling, interpolation, and batched access to motion data.
    The library can load from individual .motion files, YAML descriptors, or pre-packaged .pt files.

    **Motion Data Stored:**

    - Rigid body positions, rotations, velocities, angular velocities
    - DOF positions and velocities
    - Contact information
    - Motion metadata (lengths, FPS, weights)

    **Example:**

    .. code-block:: python

        config = MotionLibConfig(motion_file="data/motions/walk.pt")
        motion_lib = MotionLib(config, device="cuda")
        motion_ids = motion_lib.sample_motions(num_samples=1024)
        state = motion_lib.get_motion_state(motion_ids, motion_times)
    """

    # List all the tensor fields that need to be saved/loaded
    gts: torch.Tensor
    grs: torch.Tensor
    gvs: torch.Tensor
    gavs: torch.Tensor
    dvs: torch.Tensor
    dps: torch.Tensor
    length_starts: torch.Tensor
    motion_lengths: torch.Tensor
    motion_dt: torch.Tensor
    motion_num_frames: torch.Tensor
    motion_weights: torch.Tensor
    contacts: torch.Tensor

    motion_files: Tuple[str]

    # Optional fields
    lrs: Optional[torch.Tensor] = (
        None  # maybe also has local_rigid_body_rot for interpolation, see hack below
    )

    # Get all field names defined at class level
    _fields = list(__annotations__.keys())

    def __init__(
        self,
        config: "MotionLibConfig",
        device: str = "cpu",
    ):
        """Initialize MotionLib from config.

        Creates either a populated motion library (if config.motion_file is set) or
        an empty motion library (if config.motion_file is None) following Null Object pattern.

        Args:
            config: MotionLibConfig (always required, motion_file can be None for empty)
            device: PyTorch device
        """
        super().__init__()

        self.config = config
        self.device = device

        # Handle empty motion library (Null Object pattern)
        if config.motion_file is None:
            print("Creating empty MotionLib (no motion data)")
            self._create_empty()
            return

        self.get_motion_state_use_blend = config.get_motion_state_use_blend
        self.different_motion_files_across_ranks = False

        motion_file = config.motion_file
        world_size = config.world_size

        if str(motion_file).split(".")[-1] == "pt":
            print("Loading motions from packaged file which is faster")
            motion_file = self.process_packaged_motion_file_name_multi_gpu(
                motion_file, world_size
            )
            self.load_from_file(motion_file)
        else:
            print(
                "Loading motions from yaml/npy file or Directory of motions which is slower"
            )
            self._load_motions(motion_file)

        self.motion_file = motion_file

    def _create_empty(self):
        """Create an empty motion library with no motions."""
        self.get_motion_state_use_blend = False
        self.different_motion_files_across_ranks = False
        self.motion_file = None

        # Create empty tensors
        self.gts = torch.empty(0, 0, 3, device=self.device)
        self.grs = torch.empty(0, 0, 4, device=self.device)
        self.gvs = torch.empty(0, 0, 3, device=self.device)
        self.gavs = torch.empty(0, 0, 3, device=self.device)
        self.dvs = torch.empty(0, 0, device=self.device)
        self.dps = torch.empty(0, 0, device=self.device)
        self.length_starts = torch.empty(0, dtype=torch.long, device=self.device)
        self.motion_lengths = torch.empty(0, device=self.device)
        self.motion_dt = torch.empty(0, device=self.device)
        self.motion_num_frames = torch.empty(0, dtype=torch.long, device=self.device)
        self.motion_weights = torch.empty(0, device=self.device)
        self.contacts = torch.empty(0, 0, device=self.device)
        self.motion_files = ()
        self.lrs = None

    @classmethod
    def empty(cls, device: str = "cpu"):
        """Create an empty MotionLib with no motion data.

        Factory method for creating empty motion libraries in a concise way.

        Args:
            device: PyTorch device

        Returns:
            Empty MotionLib instance
        """
        return cls(config=MotionLibConfig(motion_file=None), device=device)

    def num_motions(self):
        """Returns the number of motions in the state.

        Returns:
            int: The number of motions.
        """
        return len(self.motion_lengths)

    def get_total_length(self):
        """Returns the total length of all motions.

        Returns:
            int: The total length of all motions.
        """
        return sum(self.motion_lengths)

    def get_motion_length(self, motion_ids):
        """Returns the length of the specified motion(s).

        Args:
            motion_ids: The IDs of the motions to get the length of.

        Returns:
            Tensor: The length of the specified motion(s).
            If motion_ids is None, returns the length of all motions.
        """

        if motion_ids is None:
            return self.motion_lengths
        else:
            return self.motion_lengths[motion_ids]

    def get_motion_num_frames(self, motion_ids):
        """Returns the number of frames of the specified motion(s).

        Args:
            motion_ids: The IDs of the motions to get the number of frames of.

        Returns:
            Tensor: The number of frames of the specified motion(s).
            If motion_ids is None, returns the number of frames of all motions.
        """

        if motion_ids is None:
            return self.motion_num_frames
        else:
            return self.motion_num_frames[motion_ids]

    def process_packaged_motion_file_name_multi_gpu(self, motion_file, world_size):
        # technically, this should be in a "TaskManager" class
        # so different files are different "Tasks"
        # putting it here for now since we don't have other tasks justifying creating TaskManager class

        motion_path = Path(motion_file)
        if "slurmrank.pt" in motion_path.name:
            # Get the current rank
            rank = _get_rank()
            if rank is None:
                rank = 0

            # Replace slurmrank with the actual rank number
            # This maps rank 0 to xxx_0.pt, rank 1 to xxx_1.pt, etc.
            motion_file = motion_file.replace("slurmrank.pt", f"{rank}.pt")

            self.different_motion_files_across_ranks = True
            print(f"Rank {rank} loading motion file: {motion_file}")

        return motion_file

    def _calc_frame_blend_from_id_and_time(self, motion_ids, motion_times):
        motion_len = self.motion_lengths[motion_ids]
        motion_times = motion_times.clip(min=0).clip(
            max=motion_len
        )  # Making sure time is in bounds
        num_frames = self.motion_num_frames[motion_ids]
        dt = self.motion_dt[motion_ids]

        return calc_frame_blend(motion_times, motion_len, num_frames, dt)

    def _calc_closest_frame(self, motion_ids, motion_times):
        motion_len = self.motion_lengths[motion_ids]
        motion_times = motion_times.clip(min=0).clip(
            max=motion_len
        )  # Making sure time is in bounds
        num_frames = self.motion_num_frames[motion_ids]
        frame_idx = torch.round(motion_times / motion_len * (num_frames - 1)).long()
        return frame_idx

    def get_motion_state(
        self, motion_ids, motion_times, joint_3d_format="exp_map"
    ) -> RobotState:
        frame_idx0, frame_idx1, blend = self._calc_frame_blend_from_id_and_time(
            motion_ids, motion_times
        )

        motion_state_0: RobotState = self.get_motion_state_exact_frame(
            motion_ids, frame_idx0
        )

        motion_state_1: RobotState = self.get_motion_state_exact_frame(
            motion_ids, frame_idx1
        )

        pos_keys = [
            "rigid_body_pos",
            "rigid_body_vel",
            "rigid_body_ang_vel",
            "dof_vel",
        ]

        rot_keys = ["rigid_body_rot"]

        for key in pos_keys:
            motion_state_0[key] = interpolate_pos(
                motion_state_0[key], motion_state_1[key], blend
            )

        for key in rot_keys:
            motion_state_0[key] = interpolate_quat(
                motion_state_0[key], motion_state_1[key], blend
            )

        # TODO: HACK: assume when local_rigid_body_rot is not None, all joints are exp_map
        # will use local_rigid_body_rot for interpolation
        if motion_state_0.local_rigid_body_rot is not None:
            # lr: (num_envs, num_bodies, 4)
            lr = interpolate_quat(
                motion_state_0.local_rigid_body_rot,
                motion_state_1.local_rigid_body_rot,
                blend,
            )
            b, j, _ = lr.shape
            lr = lr[:, 1:, :].reshape(
                -1, 4
            )  # (num_envs * num_bodies - 1, 4), excluding root
            assert (
                motion_state_0.dof_pos.shape[1] == (j - 1) * 3
            ), "dof_pos shape mismatch"
            motion_state_0.dof_pos = quat_to_exp_map(lr, w_last=True).reshape(
                b, (j - 1) * 3
            )
        else:
            motion_state_0.dof_pos = interpolate_pos(
                motion_state_0.dof_pos, motion_state_1.dof_pos, blend
            )

        # Blend contacts: use OR for boolean, average for float (smoothed contacts)
        if motion_state_0.rigid_body_contacts.dtype == torch.bool:
            motion_state_0.rigid_body_contacts = (
                motion_state_0.rigid_body_contacts | motion_state_1.rigid_body_contacts
            )
        else:
            # For smoothed (float) contacts, take the average between frames
            motion_state_0.rigid_body_contacts = (
                motion_state_0.rigid_body_contacts + motion_state_1.rigid_body_contacts
            ) / 2.0

        return motion_state_0

    def get_motion_state_exact_frame(
        self,
        motion_ids,
        frame_indices,
    ) -> RobotState:
        """
        Retrieves motion states at exact frame indices without any blending.

        Args:
            motion_ids: Tensor of motion IDs to sample from
            frame_indices: Tensor of integer frame indices

        Returns:
            RobotState: The robot state at the specified frames
        """

        # Get global indices by adding offsets
        fl = frame_indices + self.length_starts[motion_ids]

        # Create a dict with keys from motion_field_mapping values
        motion_data = {}
        for lib_field, motion_attr in _motion_field_mapping.items():
            motion_data[motion_attr] = getattr(self, lib_field)[fl].clone()

        if self.lrs is not None:
            local_rigid_body_rot = self.lrs[fl].clone()
        else:
            local_rigid_body_rot = None

        # Create and return the motion state
        motion_state = RobotState.from_dict(
            motion_data, state_conversion=StateConversion.COMMON
        )
        motion_state.local_rigid_body_rot = local_rigid_body_rot
        motion_state.rigid_body_contacts = self.contacts[fl].clone()

        return motion_state

    def _load_motions(self, motion_file):
        motions = []
        motion_lengths = []
        motion_dt = []
        motion_num_frames = []

        motion_files, motion_weights = self._fetch_motion_files(motion_file)

        num_motion_files = len(motion_files)

        for f in range(num_motion_files):
            curr_file = motion_files[f]
            print(curr_file)
            print(
                "Loading {:d}/{:d} motion files: {:s}".format(
                    f + 1, num_motion_files, curr_file
                )
            )

            curr_motion = torch.load(curr_file, weights_only=False)
            curr_motion = RobotState.from_dict(
                curr_motion, state_conversion=StateConversion.COMMON
            )

            motions.append(curr_motion)
            motion_lengths.append(curr_motion.motion_length)
            motion_dt.append(curr_motion.motion_dt)
            motion_num_frames.append(curr_motion.motion_num_frames)

        # Process the motions using the field mapping
        for lib_field, motion_attr in _motion_field_mapping.items():
            tp = (
                torch.bool
                if getattr(motions[0], motion_attr).dtype == torch.bool
                else torch.float32
            )
            setattr(
                self,
                lib_field,
                torch.cat([getattr(m, motion_attr) for m in motions], dim=0).to(
                    dtype=tp, device=self.device
                ),
            )

        # optionally pack local_rigid_body_rot if exists
        if motions[0].local_rigid_body_rot is not None:
            self.lrs = torch.cat(
                [getattr(m, "local_rigid_body_rot") for m in motions], dim=0
            ).to(dtype=torch.float32, device=self.device)

        # Handle other fields that don't come directly from the motion objects
        self.motion_num_frames = torch.tensor(
            motion_num_frames, dtype=torch.long, device=self.device
        )
        lengths_shifted = self.motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)

        self.motion_weights = torch.tensor(
            motion_weights, dtype=torch.float32, device=self.device
        )

        self.motion_lengths = torch.tensor(
            motion_lengths, dtype=torch.float32, device=self.device
        )
        self.motion_dt = torch.tensor(
            motion_dt, dtype=torch.float32, device=self.device
        )

        self.motion_files = tuple(motion_files)  # for saving to packed pt file

        num_motions = len(motions)
        total_len = sum(motion_lengths)
        print(
            "Loaded {:d} motions with a total length of {:.3f}s.".format(
                num_motions, total_len
            )
        )

        return

    def _fetch_motion_files(self, motion_file):
        ext = os.path.splitext(motion_file)[1]
        if ext == ".yaml":
            dir_name = os.path.dirname(motion_file)

            motion_files = []
            motion_weights = []

            with open(os.path.join(os.getcwd(), motion_file), "r") as f:
                motion_config = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))

            for motion_entry in motion_config.motions:
                curr_file = motion_entry.file
                curr_file = os.path.join(dir_name, curr_file)
                motion_files.append(curr_file)
                motion_weights.append(motion_entry.get("weight", 1.0))

        elif ext == ".npz" or ext == ".motion":
            motion_files = [motion_file]
            motion_weights = [1.0]
        else:
            # this should be a directory of motions
            motion_path = Path(motion_file)
            assert (
                motion_path.is_dir()
            ), "Motion file must be yaml, npz, motion, or a directory"

            motion_files = [str(path) for path in motion_path.glob("*.motion")]
            assert len(motion_files) > 0, "No motion files found in directory"
            motion_weights = [1.0] * len(motion_files)

        return (
            motion_files,
            motion_weights,
        )

    def save_to_file(self, file_path):
        """
        Save the motion library to a packaged file (.pt).

        Args:
            file_path: Path to save the motion library
        """

        assert str(file_path).split(".")[-1] == "pt", "Name much ends with .pt"

        file_path = Path(file_path)

        # Create a dictionary with all required tensors
        save_data = {}

        for field in self._fields:
            if getattr(self, field) is not None:
                save_data[field] = getattr(self, field)

        # Ensure directory exists
        os.makedirs(file_path.parent, exist_ok=True)

        # Save to file
        torch.save(save_data, file_path)
        print(f"Motion library saved to {file_path}")

    def load_from_file(self, file_path):
        """
        Load the motion library from a packaged file (.pt).

        Args:
            file_path: Path to the motion library file
        """
        print(f"Loading motion library from {file_path}")
        try:
            loaded_data = torch.load(
                file_path, map_location=self.device, weights_only=False
            )

            for field in loaded_data:
                assert loaded_data[field] is not None, f"Field {field} is None"
                setattr(self, field, loaded_data[field])

        except Exception as e:
            print(f"Error loading motion library: {e}")
            raise

    def smooth_contacts(self, window_size: int):
        """
        Smooth binary contact labels using a moving average filter.

        This method validates that contacts are binary, then applies a uniform
        moving average convolution to produce smoothed contact probabilities in [0, 1].
        The smoothing is applied in-place, replacing self.contacts.

        IMPORTANT: Smoothing respects motion boundaries - each motion is smoothed
        independently to avoid artifacts from one motion bleeding into another.

        Args:
            window_size: Size of the moving average window (must be positive odd number)

        Raises:
            ValueError: If contacts are not binary or window_size is invalid
        """
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")

        if window_size % 2 == 0:
            raise ValueError(
                f"window_size must be odd for symmetric smoothing, got {window_size}"
            )

        if not hasattr(self, "contacts") or self.contacts is None:
            print("Warning: No contacts to smooth (contacts field is None)")
            return

        # Validate that contacts are binary (0/1 or boolean)
        if self.contacts.dtype == torch.bool:
            # Boolean tensors are already binary, convert to float for smoothing
            self.contacts = self.contacts.float()
        else:
            # For non-boolean tensors, validate they contain only 0 and 1
            contacts_rounded = self.contacts.round()
            is_binary = torch.allclose(self.contacts, contacts_rounded, atol=1e-5)

            if not is_binary:
                # Find non-binary values for better error message
                non_binary_mask = ~torch.isclose(
                    self.contacts, contacts_rounded, atol=1e-5
                )
                non_binary_values = self.contacts[non_binary_mask]
                raise ValueError(
                    f"Contact labels must be binary (0 or 1) before smoothing. "
                    f"Found {non_binary_mask.sum().item()} non-binary values. "
                    f"Sample non-binary values: {non_binary_values[:5].tolist()}"
                )

        print(f"Smoothing contact labels with window size {window_size}...")

        # contacts shape: [total_frames, num_bodies]
        total_frames, num_bodies = self.contacts.shape
        num_motions = self.num_motions()

        # Create uniform kernel for moving average
        kernel = (
            torch.ones(1, 1, window_size, device=self.device, dtype=torch.float32)
            / window_size
        )
        padding = window_size // 2

        # Smooth each motion independently to respect motion boundaries
        smoothed_contacts = torch.zeros_like(self.contacts, dtype=torch.float32)

        for motion_idx in range(num_motions):
            # Get the range for this motion
            start_idx = self.length_starts[motion_idx].item()
            num_frames = self.motion_num_frames[motion_idx].item()
            end_idx = start_idx + num_frames

            # Extract contacts for this motion: [num_frames, num_bodies]
            motion_contacts = self.contacts[start_idx:end_idx].float()

            # Reshape for conv1d: [num_bodies, 1, num_frames]
            contacts_for_conv = motion_contacts.t().unsqueeze(1)

            # Manually apply replicate padding (functional conv1d doesn't support padding_mode)
            padded_contacts = torch.nn.functional.pad(
                contacts_for_conv,
                (padding, padding),  # pad left and right
                mode="replicate",
            )

            # Apply 1D convolution (no padding needed since we already padded)
            smoothed_motion = torch.nn.functional.conv1d(
                padded_contacts, kernel, padding=0
            )

            # Reshape back to [num_frames, num_bodies] and store
            smoothed_contacts[start_idx:end_idx] = smoothed_motion.squeeze(1).t()

        # Replace contacts with smoothed version
        self.contacts = smoothed_contacts

        # Ensure values stay in [0, 1] (they should already, but clamp for numerical stability)
        self.contacts = torch.clamp(self.contacts, 0.0, 1.0)

        print(
            f"Contact smoothing complete for {num_motions} motions. Contacts are now float values in [0, 1]."
        )

    def translate_all_motions_to_origin(self, target_xy: Optional[torch.Tensor] = None):
        """
        Translate all motions so their first frames start at the specified x,y position.

        Args:
            target_xy: Target x,y position as tensor [2]. If None, uses (0.0, 0.0)
        """
        if target_xy is None:
            target_xy = torch.zeros(2, device=self.device)

            # Process each motion individually
        for motion_idx in range(self.num_motions()):
            # Get the range for this motion (convert tensors to integers)
            start_idx = self.length_starts[motion_idx].item()
            length = self.motion_num_frames[
                motion_idx
            ].item()  # Use motion_num_frames instead of motion_lengths
            end_idx = start_idx + length

            # Get the first frame's root position for this motion
            first_frame_root_pos = self.gts[start_idx, 0, :]  # [3] - root body position
            current_xy = first_frame_root_pos[:2]  # [2]

            # Calculate translation needed (only in x,y, keep z unchanged)
            translation_xy = target_xy - current_xy  # [2]
            translation = torch.zeros(3, device=self.device)
            translation[:2] = translation_xy

            self.gts[start_idx:end_idx, :, :] += translation.reshape(1, 1, 3)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Motion Library utilities")
    parser.add_argument(
        "--motion-path", type=str, default="", help="Path to motion file (.yaml, .motion, .pt) or directory"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="motion_lib.pt",
        help="Output file path for saving motion library",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for processing (cpu or cuda)",
    )

    args = parser.parse_args()

    motion_file = args.motion_path

    # If the file is a YAML, verify motion files are accessible relative to YAML location
    if motion_file.endswith(".yaml"):
        yaml_dir = Path(motion_file).parent.resolve()
        with open(motion_file, "r") as f:
            motion_config = yaml.load(f, Loader=yaml.SafeLoader)
        
        motions = motion_config.get("motions", [])
        if motions and "file" in motions[0]:
            first_motion_path = yaml_dir / motions[0]["file"]
            if not first_motion_path.exists():
                raise FileNotFoundError(
                    f"Motion file not found: {first_motion_path}\n"
                    f"The YAML references '{motions[0]['file']}' but it doesn't exist "
                    f"relative to the YAML directory ({yaml_dir}).\n"
                    f"Did you forget to copy the YAML file to the motion directory?"
                )

    # Create and save motion library
    motion_lib = MotionLib(
        config=MotionLibConfig(motion_file=motion_file), device=args.device
    )
    motion_lib.save_to_file(args.output_file)
