# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List, Tuple
from easydict import EasyDict

import numpy as np
import torch
import yaml
from lightning_fabric.utilities.rank_zero import _get_rank
from torch import Tensor, nn

from isaac_utils import rotations, torch_utils
from phys_anim.utils.device_dtype_mixin import DeviceDtypeModuleMixin
from poselib.core.rotation3d import quat_angle_axis, quat_inverse, quat_mul_norm
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState


@dataclass
class MotionState:
    root_pos: Tensor
    root_rot: Tensor
    dof_pos: Tensor
    root_vel: Tensor
    root_ang_vel: Tensor
    dof_vel: Tensor
    key_body_pos: Tensor
    rb_pos: Tensor
    rb_rot: Tensor
    local_rot: Tensor
    rb_vel: Tensor
    rb_ang_vel: Tensor


class LoadedMotions(nn.Module):
    def __init__(
        self,
        motions: Tuple[SkeletonMotion],
        motion_lengths: Tensor,
        motion_weights: Tensor,
        motion_timings: Tensor,
        motion_fps: Tensor,
        motion_dt: Tensor,
        motion_num_frames: Tensor,
        motion_files: Tuple[str],
        sub_motion_to_motion: Tensor,
        ref_respawn_offsets: Tensor,
        text_embeddings: Tensor = None,
        has_text_embeddings: Tensor = None,
        supported_scene_ids: List[List[str]] = None,
        **kwargs,  # Catch some nn.Module arguments that aren't needed
    ):
        super().__init__()
        self.motions = motions
        self.motion_files = motion_files
        self.register_buffer("motion_lengths", motion_lengths, persistent=False)
        self.register_buffer("motion_weights", motion_weights, persistent=False)
        self.register_buffer("motion_timings", motion_timings, persistent=False)
        self.register_buffer("motion_fps", motion_fps, persistent=False)
        self.register_buffer("motion_dt", motion_dt, persistent=False)
        self.register_buffer("motion_num_frames", motion_num_frames, persistent=False)
        self.register_buffer(
            "sub_motion_to_motion", sub_motion_to_motion, persistent=False
        )
        self.register_buffer(
            "ref_respawn_offsets", ref_respawn_offsets, persistent=False
        )
        if text_embeddings is None:
            text_embeddings = torch.zeros(len(motions), 3, 512, dtype=torch.float32)
            has_text_embeddings = torch.zeros(len(motions), dtype=torch.bool)
        self.register_buffer("text_embeddings", text_embeddings, persistent=False)
        self.register_buffer(
            "has_text_embeddings", has_text_embeddings, persistent=False
        )
        if supported_scene_ids is None:
            supported_scene_ids = [None for _ in range(len(motions))]
        self.supported_scene_ids = supported_scene_ids


class MotionLib(DeviceDtypeModuleMixin):
    gts: Tensor
    grs: Tensor
    lrs: Tensor
    gvs: Tensor
    gavs: Tensor
    grvs: Tensor
    gravs: Tensor
    dvs: Tensor
    length_starts: Tensor
    motion_ids: Tensor
    key_body_ids: Tensor

    def __init__(
        self,
        motion_file,
        dof_body_ids,
        dof_offsets,
        key_body_ids,
        device="cpu",
        ref_height_adjust: float = 0,
        target_frame_rate: int = 30,
        create_text_embeddings: bool = False,
        spawned_scene_ids: List[str] = None,
        fix_motion_heights: bool = True,
        skeleton_tree: Any = None,
        local_rot_conversion: Tensor = None,
        w_last: bool = True,
    ):
        super().__init__()
        self.w_last = w_last
        self.fix_heights = fix_motion_heights
        self.skeleton_tree = skeleton_tree
        self.create_text_embeddings = create_text_embeddings
        self.dof_body_ids = dof_body_ids
        self.dof_offsets = dof_offsets
        self.num_dof = dof_offsets[-1]
        self.ref_height_adjust = ref_height_adjust
        self.local_rot_conversion = local_rot_conversion

        self.register_buffer(
            "key_body_ids",
            torch.tensor(key_body_ids, dtype=torch.long, device=device),
            persistent=False,
        )

        if str(motion_file).split(".")[-1] in ["yaml", "npy", "npz", "np"]:
            print("Loading motions from yaml/npy file")
            self._load_motions(motion_file, target_frame_rate)
        else:
            rank = _get_rank()
            if rank is None:
                rank = 0
            # This is used for large motion files that are split across multiple GPUs
            motion_file = motion_file.replace("_slurmrank", f"_{rank}")
            print(f"Loading motions from state file: {motion_file}")

            with open(motion_file, "rb") as file:
                state: LoadedMotions = torch.load(file, map_location="cpu")

            # Create LoadedMotions instance with loaded state dict
            # We re-create to enable backwards compatibility. This allows LoadedMotions class to accept "None" values and set defaults if needed.
            state_dict = {
                **vars(state),
                **{k: v for k, v in state._buffers.items() if v is not None},
            }
            self.state = LoadedMotions(**state_dict)

        motions = self.state.motions
        self.register_buffer(
            "gts",
            torch.cat([m.global_translation for m in motions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "grs",
            torch.cat([m.global_rotation for m in motions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "lrs",
            torch.cat([m.local_rotation for m in motions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "grvs",
            torch.cat([m.global_root_velocity for m in motions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "gravs",
            torch.cat([m.global_root_angular_velocity for m in motions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "gavs",
            torch.cat([m.global_angular_velocity for m in motions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "gvs",
            torch.cat([m.global_velocity for m in motions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "dvs",
            torch.cat([m.dof_vels for m in motions], dim=0).to(
                device=device, dtype=torch.float32
            ),
            persistent=False,
        )

        lengths = self.state.motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.register_buffer(
            "length_starts", lengths_shifted.cumsum(0), persistent=False
        )

        self.register_buffer(
            "motion_ids",
            torch.arange(
                len(self.state.motions), dtype=torch.long, device=self._device
            ),
            persistent=False,
        )

        scenes_per_motion, motion_to_scene_ids = self.parse_scenes(spawned_scene_ids)

        self.register_buffer(
            "scenes_per_motion",
            torch.tensor(scenes_per_motion, device=self._device, dtype=torch.long),
            persistent=False,
        )

        self.register_buffer(
            "motion_to_scene_ids",
            torch.tensor(motion_to_scene_ids, device=self._device, dtype=torch.long),
            persistent=False,
        )

        self.to(device)

    def num_motions(self):
        """Returns the number of motions in the state.

        Returns:
            int: The number of motions.
        """
        return len(self.state.motions)

    def num_sub_motions(self):
        """Returns the number of sub-motions in the state.

        A sub-motion is a segment or a part of a larger motion sequence.
        In the context of this code, a motion can be divided into multiple sub-motions,
        each representing a smaller portion of the overall motion.
        These sub-motions are used to manage and manipulate parts of the motion sequence
        independently, allowing for more granular control and analysis of the motion data.

        Returns:
            int: The number of sub-motions.
        """
        return self.state.motion_weights.shape[0]

    def get_total_length(self):
        """Returns the total length of all motions.

        Returns:
            int: The total length of all motions.
        """
        return sum(self.state.motion_lengths)

    def get_total_trainable_length(self):
        """Returns the total trainable length of all motions.

        The total trainable length is calculated by summing the differences
        between the end and start times of each motion timing.

        Returns:
            int: The total trainable length of all motions.
        """
        return sum(self.state.motion_timings[:, 1] - self.state.motion_timings[:, 0])

    def get_motion(self, motion_id):
        return self.state.motions[motion_id]

    def sample_motions(self, n, valid_mask=None):
        if valid_mask is not None:
            weights = self.state.motion_weights.clone()
            weights[~valid_mask] = 0
        else:
            weights = self.state.motion_weights

        sub_motion_ids = torch.multinomial(weights, num_samples=n, replacement=True)

        return sub_motion_ids

    def sample_other_motions(self, already_chosen_ids: Tensor) -> Tensor:
        """Samples other motions that are not in the already chosen IDs.

        Args:
            already_chosen_ids (Tensor): A tensor containing the IDs of motions that have already been chosen.

        Returns:
            Tensor: A tensor containing the IDs of the sampled motions that are not in the already chosen IDs.
        """
        n = already_chosen_ids.shape[0]
        motion_weights = self.state.motion_weights.unsqueeze(0).tile([n, 1])
        motion_weights = motion_weights.scatter(
            1, already_chosen_ids.unsqueeze(-1), torch.zeros_like(motion_weights)
        )
        sub_motion_ids = torch.multinomial(motion_weights, num_samples=1).squeeze(-1)
        return sub_motion_ids

    def sample_text_embeddings(self, sub_motion_ids: Tensor) -> Tensor:
        """Samples text embeddings for the given sub-motion IDs.

        Args:
            sub_motion_ids (Tensor): A tensor containing the IDs of the sub-motions.

        Returns:
            Tensor: A tensor containing the sampled text embeddings for the given sub-motion IDs.
        """
        if hasattr(self.state, "text_embeddings"):
            indices = torch.randint(
                0, 3, (sub_motion_ids.shape[0],), device=self.device
            )
            return self.state.text_embeddings[sub_motion_ids, indices]
        return 0

    def sample_time(self, sub_motion_ids, max_time=None, truncate_time=None):
        phase = torch.rand(sub_motion_ids.shape, device=self.device)

        motion_len = (
            self.state.motion_timings[sub_motion_ids, 1]
            - self.state.motion_timings[sub_motion_ids, 0]
        )
        if max_time is not None:
            motion_len = torch.clamp(
                motion_len,
                max=max_time,
            )

        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len -= truncate_time
            assert torch.all(motion_len >= 0)

        motion_time = phase * motion_len
        return motion_time + self.state.motion_timings[sub_motion_ids, 0]

    def get_sub_motion_length(self, sub_motion_ids):
        return (
            self.state.motion_timings[sub_motion_ids, 1]
            - self.state.motion_timings[sub_motion_ids, 0]
        )

    def get_motion_length(self, sub_motion_ids):
        motion_ids = self.state.sub_motion_to_motion[sub_motion_ids]
        return self.state.motion_lengths[motion_ids]

    def get_mimic_motion_state(
        self, sub_motion_ids, motion_times, joint_3d_format="exp_map"
    ) -> MotionState:
        motion_ids = self.state.sub_motion_to_motion[sub_motion_ids]

        motion_len = self.state.motion_lengths[motion_ids]
        motion_times = motion_times.clip(min=0).clip(
            max=motion_len
        )  # Making sure time is in bounds

        num_frames = self.state.motion_num_frames[motion_ids]
        dt = self.state.motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(
            motion_times, motion_len, num_frames, dt
        )

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        global_translation0 = self.gts[f0l]
        global_translation1 = self.gts[f1l]

        global_rotation0 = self.grs[f0l]
        global_rotation1 = self.grs[f1l]

        local_rotation0 = self.lrs[f0l]
        local_rotation1 = self.lrs[f1l]

        global_vel0 = self.gvs[f0l]
        global_vel1 = self.gvs[f1l]

        global_ang_vel0 = self.gavs[f0l]
        global_ang_vel1 = self.gavs[f1l]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        blend = blend.unsqueeze(-1)
        blend_exp = blend.unsqueeze(-1)

        global_translation: Tensor = (
            1.0 - blend_exp
        ) * global_translation0 + blend_exp * global_translation1
        global_rotation: Tensor = torch_utils.slerp(
            global_rotation0, global_rotation1, blend_exp
        )

        local_rotation: Tensor = torch_utils.slerp(
            local_rotation0, local_rotation1, blend_exp
        )

        if hasattr(self, "dof_pos"):  # H1 joints
            dof_pos = (1.0 - blend) * self.dof_pos[f0l] + blend * self.dof_pos[f1l]
        else:
            dof_pos: Tensor = self._local_rotation_to_dof(
                local_rotation, joint_3d_format
            )

        global_vel = (1.0 - blend_exp) * global_vel0 + blend_exp * global_vel1
        global_ang_vel = (
            1.0 - blend_exp
        ) * global_ang_vel0 + blend_exp * global_ang_vel1
        dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1

        global_translation[:, :, 2] += self.ref_height_adjust

        motion_state = MotionState(
            root_pos=None,
            root_rot=None,
            root_vel=None,
            root_ang_vel=None,
            key_body_pos=None,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            rb_pos=global_translation,
            rb_rot=global_rotation,
            local_rot=local_rotation,
            rb_vel=global_vel,
            rb_ang_vel=global_ang_vel,
        )

        return motion_state

    def get_motion_state(
        self, sub_motion_ids, motion_times, joint_3d_format="exp_map"
    ) -> MotionState:
        motion_ids = self.state.sub_motion_to_motion[sub_motion_ids]

        motion_len = self.state.motion_lengths[motion_ids]
        motion_times = motion_times.clip(min=0).clip(
            max=motion_len
        )  # Making sure time is in bounds

        num_frames = self.state.motion_num_frames[motion_ids]
        dt = self.state.motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(
            motion_times, motion_len, num_frames, dt
        )

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        root_pos0 = self.gts[f0l, 0]
        root_pos1 = self.gts[f1l, 0]

        root_rot0 = self.grs[f0l, 0]
        root_rot1 = self.grs[f1l, 0]

        local_rot0 = self.lrs[f0l]
        local_rot1 = self.lrs[f1l]

        root_vel0 = self.grvs[f0l]
        root_vel1 = self.grvs[f1l]

        root_ang_vel0 = self.gravs[f0l]
        root_ang_vel1 = self.gravs[f1l]

        global_vel0 = self.gvs[f0l]
        global_vel1 = self.gvs[f1l]

        global_ang_vel0 = self.gavs[f0l]
        global_ang_vel1 = self.gavs[f1l]

        key_body_pos0 = self.gts[f0l.unsqueeze(-1), self.key_body_ids.unsqueeze(0)]
        key_body_pos1 = self.gts[f1l.unsqueeze(-1), self.key_body_ids.unsqueeze(0)]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        rb_pos0 = self.gts[f0l]
        rb_pos1 = self.gts[f1l]

        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]

        vals = [
            root_pos0,
            root_pos1,
            local_rot0,
            local_rot1,
            root_vel0,
            root_vel1,
            root_ang_vel0,
            root_ang_vel1,
            global_vel0,
            global_vel1,
            global_ang_vel0,
            global_ang_vel1,
            dof_vel0,
            dof_vel1,
            key_body_pos0,
            key_body_pos1,
            rb_pos0,
            rb_pos1,
            rb_rot0,
            rb_rot1,
        ]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        root_pos: Tensor = (1.0 - blend) * root_pos0 + blend * root_pos1
        root_pos[:, 2] += self.ref_height_adjust

        root_rot: Tensor = torch_utils.slerp(root_rot0, root_rot1, blend)

        blend_exp = blend.unsqueeze(-1)
        key_body_pos = (1.0 - blend_exp) * key_body_pos0 + blend_exp * key_body_pos1
        key_body_pos[:, :, 2] += self.ref_height_adjust

        local_rot = torch_utils.slerp(
            local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1)
        )

        if hasattr(self, "dof_pos"):  # H1 joints
            dof_pos = (1.0 - blend) * self.dof_pos[f0l] + blend * self.dof_pos[f1l]
        else:
            dof_pos: Tensor = self._local_rotation_to_dof(local_rot, joint_3d_format)

        root_vel = (1.0 - blend) * root_vel0 + blend * root_vel1
        root_ang_vel = (1.0 - blend) * root_ang_vel0 + blend * root_ang_vel1
        dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1
        rb_pos = (1.0 - blend_exp) * rb_pos0 + blend_exp * rb_pos1
        rb_pos[:, :, 2] += self.ref_height_adjust
        rb_rot = torch_utils.slerp(rb_rot0, rb_rot1, blend_exp)
        global_vel = (1.0 - blend_exp) * global_vel0 + blend_exp * global_vel1
        global_ang_vel = (
            1.0 - blend_exp
        ) * global_ang_vel0 + blend_exp * global_ang_vel1

        motion_state = MotionState(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            key_body_pos=key_body_pos,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            local_rot=local_rot,
            rb_pos=rb_pos,
            rb_rot=rb_rot,
            rb_vel=global_vel,
            rb_ang_vel=global_ang_vel,
        )

        return motion_state

    @staticmethod
    def _load_motion_file(motion_file):
        return SkeletonMotion.from_file(motion_file)

    def _load_motions(self, motion_file, target_frame_rate):
        if self.create_text_embeddings:
            from transformers import AutoTokenizer, XCLIPTextModel

            model = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")

        motions = []
        motion_lengths = []
        motion_dt = []
        motion_num_frames = []
        text_embeddings = []
        has_text_embeddings = []
        (
            motion_files,
            motion_weights,
            motion_timings,
            motion_fpses,
            sub_motion_to_motion,
            ref_respawn_offsets,
            motion_labels,
            supported_scene_ids,
        ) = self._fetch_motion_files(motion_file)

        num_motion_files = len(motion_files)

        for f in range(num_motion_files):
            curr_file = motion_files[f]

            print(
                "Loading {:d}/{:d} motion files: {:s}".format(
                    f + 1, num_motion_files, curr_file
                )
            )
            curr_motion = self._load_motion_file(curr_file)

            curr_motion = fix_motion_fps(
                curr_motion,
                motion_fpses[f],
                target_frame_rate,
                self.skeleton_tree,
            )
            motion_fpses[f] = float(curr_motion.fps)

            if self.fix_heights:
                curr_motion = self.fix_motion_heights(curr_motion, self.skeleton_tree)

            curr_dt = 1.0 / motion_fpses[f]

            num_frames = curr_motion.global_translation.shape[0]
            curr_len = 1.0 / motion_fpses[f] * (num_frames - 1)

            motion_dt.append(curr_dt)
            motion_num_frames.append(num_frames)

            curr_dof_vels = self._compute_motion_dof_vels(curr_motion)
            curr_motion.dof_vels = curr_dof_vels

            motions.append(curr_motion)
            motion_lengths.append(curr_len)

        num_sub_motions = len(sub_motion_to_motion)

        for f in range(num_sub_motions):
            # Incase start/end weren't provided, set to (0, motion_length)
            motion_f = sub_motion_to_motion[f]
            if motion_timings[f][1] == -1:
                motion_timings[f][1] = motion_lengths[motion_f]

            motion_timings[f][1] = min(
                motion_timings[f][1], motion_lengths[motion_f]
            )  # CT hack: fix small timing differences

            assert (
                motion_timings[f][0] < motion_timings[f][1]
            ), f"Motion start {motion_timings[f][0]} >= motion end {motion_timings[f][1]} in motion {motion_f}"

            if self.create_text_embeddings and motion_labels[f][0] != "":
                with torch.inference_mode():
                    inputs = tokenizer(
                        motion_labels[f],
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    )
                    outputs = model(**inputs)
                    pooled_output = outputs.pooler_output  # pooled (EOS token) states
                    text_embeddings.append(pooled_output)  # should be [3, 512]
                    has_text_embeddings.append(True)
            else:
                text_embeddings.append(
                    torch.zeros((3, 512), dtype=torch.float32)
                )  # just hold something temporary
                has_text_embeddings.append(False)

        motion_lengths = torch.tensor(
            motion_lengths, device=self._device, dtype=torch.float32
        )

        motion_weights = torch.tensor(
            motion_weights, dtype=torch.float32, device=self._device
        )
        motion_weights /= motion_weights.sum()

        motion_timings = torch.tensor(
            motion_timings, dtype=torch.float32, device=self._device
        )

        sub_motion_to_motion = torch.tensor(
            sub_motion_to_motion, dtype=torch.long, device=self._device
        )

        ref_respawn_offsets = torch.tensor(
            ref_respawn_offsets, dtype=torch.float32, device=self._device
        )

        motion_fpses = torch.tensor(
            motion_fpses, device=self._device, dtype=torch.float32
        )
        motion_dt = torch.tensor(motion_dt, device=self._device, dtype=torch.float32)
        motion_num_frames = torch.tensor(motion_num_frames, device=self._device)

        text_embeddings = torch.stack(text_embeddings).detach().to(device=self._device)
        has_text_embeddings = torch.tensor(
            has_text_embeddings, dtype=torch.bool, device=self._device
        )

        self.state = LoadedMotions(
            motions=tuple(motions),
            motion_lengths=motion_lengths,
            motion_weights=motion_weights,
            motion_timings=motion_timings,
            motion_fps=motion_fpses,
            motion_dt=motion_dt,
            motion_num_frames=motion_num_frames,
            motion_files=tuple(motion_files),
            sub_motion_to_motion=sub_motion_to_motion,
            ref_respawn_offsets=ref_respawn_offsets,
            text_embeddings=text_embeddings,
            has_text_embeddings=has_text_embeddings,
            supported_scene_ids=supported_scene_ids,
        )

        num_motions = self.num_motions()
        total_len = self.get_total_length()

        print(
            "Loaded {:d} motions with a total length of {:.3f}s.".format(
                num_motions, total_len
            )
        )

        num_sub_motions = self.num_sub_motions()
        total_trainable_len = self.get_total_trainable_length()

        print(
            "Loaded {:d} sub motions with a total trainable length of {:.3f}s.".format(
                num_sub_motions, total_trainable_len
            )
        )

    def _fetch_motion_files(self, motion_file):
        ext = os.path.splitext(motion_file)[1]
        if ext == ".yaml":
            dir_name = os.path.dirname(motion_file)
            motion_files = []
            sub_motion_to_motion = []
            ref_respawn_offsets = []
            motion_weights = []
            motion_timings = []
            motion_fpses = []
            motion_labels = []
            supported_scene_ids = []
            with open(os.path.join(os.getcwd(), motion_file), "r") as f:
                motion_config = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))

            motion_list = sorted(
                motion_config.motions,
                key=lambda x: 1e6 if "idx" not in x else int(x.idx),
            )

            motion_index = 0

            for motion_id, motion_entry in enumerate(motion_list):
                curr_file = motion_entry.file
                curr_file = os.path.join(dir_name, curr_file)
                motion_files.append(curr_file)
                motion_fpses.append(motion_entry.get("fps", None))

                if "sub_motions" not in motion_entry:
                    motion_entry.sub_motions = [deepcopy(motion_entry)]
                    motion_entry.sub_motions[0].idx = motion_index

                for sub_motion in sorted(
                    motion_entry.sub_motions, key=lambda x: int(x.idx)
                ):
                    curr_weight = sub_motion.weight
                    assert curr_weight >= 0

                    assert motion_index == sub_motion.idx

                    motion_weights.append(curr_weight)

                    sub_motion_to_motion.append(motion_id)

                    ref_respawn_offset = sub_motion.get("ref_respawn_offset", 0)
                    ref_respawn_offsets.append(ref_respawn_offset)

                    if "timings" in sub_motion:
                        curr_timing = sub_motion.timings
                        start = curr_timing.start
                        end = curr_timing.end
                    else:
                        start = 0
                        end = -1

                    motion_timings.append([start, end])

                    sub_motion_labels = []
                    if "labels" in sub_motion:
                        # We assume 3 labels for each motion.
                        # If there are fewer than 3 labels, the last label is repeated to fill the list.
                        # If there are no labels, an empty string is used as the label.
                        for label in sub_motion.labels:
                            sub_motion_labels.append(label)
                            if len(sub_motion_labels) == 3:
                                break
                        if len(sub_motion_labels) == 0:
                            sub_motion_labels.append("")
                        while len(sub_motion_labels) < 3:
                            sub_motion_labels.append(sub_motion_labels[-1])
                    else:
                        sub_motion_labels.append("")
                        sub_motion_labels.append("")
                        sub_motion_labels.append("")

                    motion_labels.append(sub_motion_labels)

                    if "supported_scenes" in sub_motion:
                        supported_scene_ids.append(sub_motion.supported_scenes)
                    else:
                        supported_scene_ids.append(None)

                    motion_index += 1
        else:
            motion_files = [motion_file]
            motion_weights = [1.0]
            motion_timings = [[0, -1]]
            motion_fpses = [None]
            sub_motion_to_motion = [0]
            ref_respawn_offsets = [0]
            motion_labels = [["", "", ""]]
            supported_scene_ids = [None]
        return (
            motion_files,
            motion_weights,
            motion_timings,
            motion_fpses,
            sub_motion_to_motion,
            ref_respawn_offsets,
            motion_labels,
            supported_scene_ids,
        )

    def _calc_frame_blend(self, time, len, num_frames, dt):
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt

        return frame_idx0, frame_idx1, blend

    def _get_num_bodies(self):
        motion = self.get_motion(0)
        num_bodies = motion.num_joints
        return num_bodies

    def _compute_motion_dof_vels(self, motion: SkeletonMotion):
        num_frames = motion.global_translation.shape[0]
        dt = 1.0 / motion.fps
        dof_vels = []

        for f in range(num_frames - 1):
            local_rot0 = motion.local_rotation[f]
            local_rot1 = motion.local_rotation[f + 1]
            frame_dof_vel = self._local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
            dof_vels.append(frame_dof_vel)

        dof_vels.append(dof_vels[-1])
        dof_vels = torch.stack(dof_vels, dim=0)

        return dof_vels

    # jp hack
    # get rid of this ASAP, need a proper way of projecting from max coords to reduced coords
    def _local_rotation_to_dof(self, local_rot, joint_3d_format):
        body_ids = self.dof_body_ids
        dof_offsets = self.dof_offsets

        n = local_rot.shape[0]
        dof_pos = torch.zeros((n, self.num_dof), dtype=torch.float, device=self._device)

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if joint_size == 3:
                joint_q = local_rot[:, body_id]
                if joint_3d_format == "exp_map":
                    formatted_joint = torch_utils.quat_to_exp_map(joint_q, w_last=True)
                elif joint_3d_format == "xyz":
                    x, y, z = rotations.get_euler_xyz(joint_q, w_last=True)
                    formatted_joint = torch.stack([x, y, z], dim=-1)
                else:
                    raise ValueError(f"Unknown 3d format '{joint_3d_format}'")

                dof_pos[:, joint_offset : (joint_offset + joint_size)] = formatted_joint
            elif joint_size == 1:
                joint_q = local_rot[:, body_id]
                joint_theta, joint_axis = torch_utils.quat_to_angle_axis(
                    joint_q, w_last=True
                )
                joint_theta = (
                    joint_theta * joint_axis[..., 1]
                )  # assume joint is always along y axis

                joint_theta = rotations.normalize_angle(joint_theta)
                dof_pos[:, joint_offset] = joint_theta

            else:
                print("Unsupported joint type")
                assert False

        return dof_pos

    def _local_rotation_to_dof_vel(self, local_rot0, local_rot1, dt):
        body_ids = self.dof_body_ids
        dof_offsets = self.dof_offsets

        dof_vel = torch.zeros([self.num_dof], device=self._device)

        diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1)
        diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
        local_vel = diff_axis * diff_angle.unsqueeze(-1) / dt
        local_vel = local_vel

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if joint_size == 3:
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset : (joint_offset + joint_size)] = joint_vel

            elif joint_size == 1:
                assert joint_size == 1
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset] = joint_vel[
                    1
                ]  # assume joint is always along y axis

            else:
                print("Unsupported joint type")
                assert False

        return dof_vel

    def parse_scenes(self, spawned_scene_ids):
        # If motions may have supported scenes, create the mapping to allow sampling scenes for motions.
        motion_to_scene_ids = []
        scenes_per_motion = []
        if hasattr(self.state, "supported_scene_ids") and spawned_scene_ids is not None:

            def indices(lst, element):
                result = []
                offset = -1
                while True:
                    try:
                        offset = lst.index(element, offset + 1)
                    except ValueError:
                        return result
                    result.append(offset)

            max_num_scenes = max(
                max(
                    [
                        len(scene_ids) if scene_ids is not None else 0
                        for scene_ids in self.state.supported_scene_ids
                    ]
                ),
                len(spawned_scene_ids),
            )

            for i in range(len(self.state.supported_scene_ids)):
                if self.state.supported_scene_ids[i] is None:
                    motion_to_scene_ids.append([-1] * max_num_scenes)
                    scenes_per_motion.append(-1)
                else:
                    all_scene_ids = []
                    for scene_id in self.state.supported_scene_ids[i]:
                        if scene_id in spawned_scene_ids:
                            # store all indices that match, multiple options may exist
                            scene_indices = indices(spawned_scene_ids, scene_id)
                            for scene_index in scene_indices:
                                all_scene_ids.append(scene_index)

                    scenes_per_motion.append(len(all_scene_ids))

                    if len(all_scene_ids) == 0:
                        all_scene_ids = [-1]
                    while len(all_scene_ids) < max_num_scenes:
                        all_scene_ids.append(-1)
                    motion_to_scene_ids.append(all_scene_ids)

        return scenes_per_motion, motion_to_scene_ids

    def sample_motions_scene_aware(
        self,
        num_motions,
        available_scenes,
        single_robot_in_scene,
        with_replacement=True,
        available_motion_mask=None,
    ):
        sampled_motions = []
        occupied_scenes = []

        if available_motion_mask is None:
            available_motion_mask = torch.ones(
                len(self.scenes_per_motion), dtype=torch.bool, device=self.device
            )

        motion_weights = self.state.motion_weights.clone()

        while len(sampled_motions) < num_motions:
            # Create a view of available motions
            for i, num_scenes in enumerate(self.scenes_per_motion):
                if num_scenes != -1 and not torch.any(
                    available_scenes[self.motion_to_scene_ids[i, :num_scenes]]
                ):
                    available_motion_mask[i] = False

            # Sample a motion based on weights
            motion_weights[~available_motion_mask] = 0
            if motion_weights.sum() == 0:
                raise ValueError("No more valid motions available")
            sampled_motion = torch.multinomial(motion_weights, num_samples=1).item()
            sampled_motions.append(sampled_motion)

            if not with_replacement:
                available_motion_mask[sampled_motion] = False

            # Sample a scene for the motion if needed
            if self.scenes_per_motion[sampled_motion] != -1:
                num_scenes = self.scenes_per_motion[sampled_motion]
                available_scene_mask = available_scenes[
                    self.motion_to_scene_ids[sampled_motion, :num_scenes]
                ]
                valid_scenes = self.motion_to_scene_ids[sampled_motion, :num_scenes][
                    available_scene_mask
                ]
                if valid_scenes.numel() > 0:
                    scene = valid_scenes[
                        torch.randint(0, valid_scenes.numel(), (1,)).item()
                    ]
                    occupied_scenes.append(scene)
                    if single_robot_in_scene[scene]:
                        available_scenes[scene] = False
                else:
                    raise ValueError("No more valid scenes available")
            else:
                occupied_scenes.append(-1)

        return torch.tensor(
            sampled_motions, device=self.device, dtype=torch.long
        ), torch.tensor(occupied_scenes, device=self.device, dtype=torch.long)

    @staticmethod
    def fix_motion_heights(motion, skeleton_tree):
        if skeleton_tree is None:
            if hasattr(motion, "skeleton_tree"):
                skeleton_tree = motion.skeleton_tree
        body_heights = motion.global_translation[..., 2]
        min_height = body_heights.min()

        if skeleton_tree is None:
            motion.global_translation[..., 2] -= min_height
            return motion

        root_translation = motion.root_translation
        root_translation[:, 2] -= min_height

        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            motion.global_rotation,
            root_translation,
            is_local=False,
        )

        new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=motion.fps)

        return new_motion


def fix_motion_fps(
    motion,
    orig_fps,
    target_frame_rate,
    skeleton_tree,
):
    if skeleton_tree is None:
        if hasattr(motion, "skeleton_tree"):
            skeleton_tree = motion.skeleton_tree
        else:
            return motion

    if orig_fps is None:
        orig_fps = motion.fps

    skip = int(np.round(orig_fps / target_frame_rate))

    lr = motion.local_rotation[::skip]
    rt = motion.root_translation[::skip]

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        lr,
        rt,
        is_local=True,
    )
    new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_frame_rate)

    return new_motion
