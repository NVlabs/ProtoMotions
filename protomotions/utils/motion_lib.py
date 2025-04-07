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
from typing import Any, Tuple
from easydict import EasyDict

import numpy as np
import torch
import yaml
from lightning_fabric.utilities.rank_zero import _get_rank
from torch import Tensor, nn

from isaac_utils import rotations, torch_utils
from protomotions.simulator.base_simulator.config import RobotConfig
from protomotions.simulator.base_simulator.robot_state import RobotState
from protomotions.utils.device_dtype_mixin import DeviceDtypeModuleMixin
from poselib.core.rotation3d import quat_angle_axis, quat_inverse, quat_mul_norm
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState

# CT hack: remap phys_anim to protomotions for backwards compatibility of pre-existing motion files
import sys
import protomotions

sys.modules['phys_anim'] = protomotions


class LoadedMotions(nn.Module):
    def __init__(
            self,
            motions: Tuple[SkeletonMotion],
            motion_lengths: Tensor,
            motion_weights: Tensor,
            motion_fps: Tensor,
            motion_dt: Tensor,
            motion_num_frames: Tensor,
            motion_files: Tuple[str],
            ref_respawn_offsets: Tensor,
            text_embeddings: Tensor = None,
            has_text_embeddings: Tensor = None,
            **kwargs,  # Catch some nn.Module arguments that aren't needed
    ):
        super().__init__()
        self.motions = motions
        self.motion_files = motion_files
        self.register_buffer("motion_lengths", motion_lengths, persistent=False)
        self.register_buffer("motion_weights", motion_weights, persistent=False)
        self.register_buffer("motion_fps", motion_fps, persistent=False)
        self.register_buffer("motion_dt", motion_dt, persistent=False)
        self.register_buffer("motion_num_frames", motion_num_frames, persistent=False)
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
            robot_config: RobotConfig,
            key_body_ids,
            device="cpu",
            ref_height_adjust: float = 0,
            target_frame_rate: int = 30,
            create_text_embeddings: bool = False,
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
        self.robot_config = robot_config
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

        self.motion_file = motion_file

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

        self.to(device)

    def num_motions(self):
        """Returns the number of motions in the state.

        Returns:
            int: The number of motions.
        """
        return len(self.state.motions)

    def get_total_length(self):
        """Returns the total length of all motions.

        Returns:
            int: The total length of all motions.
        """
        return sum(self.state.motion_lengths)

    def get_motion(self, motion_id):
        return self.state.motions[motion_id]

    def sample_text_embeddings(self, motion_ids: Tensor) -> Tensor:
        """Samples text embeddings for the given motion IDs.

        Args:
            motion_ids (Tensor): A tensor containing the IDs of the motions.

        Returns:
            Tensor: A tensor containing the sampled text embeddings for the given motion IDs.
        """
        if hasattr(self.state, "text_embeddings"):
            indices = torch.randint(0, 3, (motion_ids.shape[0],), device=self.device)
            return self.state.text_embeddings[motion_ids, indices]
        return 0

    def sample_time(self, motion_ids, truncate_time=None):
        phase = torch.rand(motion_ids.shape, device=self.device)

        motion_len = self.state.motion_lengths[motion_ids]

        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len -= truncate_time
            assert torch.all(motion_len >= 0)

        motion_time = phase * motion_len
        return motion_time

    def get_motion_length(self, motion_ids):
        if motion_ids is None:
            return self.state.motion_lengths
        else:
            return self.state.motion_lengths[motion_ids]

    def get_motion_state(
            self, motion_ids, motion_times, joint_3d_format="exp_map"
    ) -> RobotState:
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

        rigid_body_pos0 = self.gts[f0l]
        rigid_body_pos1 = self.gts[f1l]

        rigid_body_rot0 = self.grs[f0l]
        rigid_body_rot1 = self.grs[f1l]

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
            rigid_body_pos0,
            rigid_body_pos1,
            rigid_body_rot0,
            rigid_body_rot1,
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

        if hasattr(self, "dof_pos"):  # H1 joints
            dof_pos = (1.0 - blend) * self.dof_pos[f0l] + blend * self.dof_pos[f1l]
        else:
            local_rot = torch_utils.slerp(
                local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1)
            )
            dof_pos: Tensor = self._local_rotation_to_dof(local_rot, joint_3d_format)

        root_vel = (1.0 - blend) * root_vel0 + blend * root_vel1
        root_ang_vel = (1.0 - blend) * root_ang_vel0 + blend * root_ang_vel1
        dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1
        rigid_body_pos = (1.0 - blend_exp) * rigid_body_pos0 + blend_exp * rigid_body_pos1
        rigid_body_pos[:, :, 2] += self.ref_height_adjust
        rigid_body_rot = torch_utils.slerp(rigid_body_rot0, rigid_body_rot1, blend_exp)
        global_vel = (1.0 - blend_exp) * global_vel0 + blend_exp * global_vel1
        global_ang_vel = (
            1.0 - blend_exp
        ) * global_ang_vel0 + blend_exp * global_ang_vel1

        motion_state = RobotState(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            key_body_pos=key_body_pos,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            rigid_body_pos=rigid_body_pos,
            rigid_body_rot=rigid_body_rot,
            rigid_body_vel=global_vel,
            rigid_body_ang_vel=global_ang_vel,
        )

        return motion_state

    @staticmethod
    def _load_motion_file(motion_file):
        return SkeletonMotion.from_file(motion_file)

    @staticmethod
    def _slice_motion_file(motion, motion_timings):
        start, end = motion_timings
        start_frame = round(start * motion.fps)
        if end == -1:
            end_frame = motion.global_translation.shape[0]
        else:
            end_frame = int(end * motion.fps)

        assert (
                start_frame < end_frame
        ), f"Motion start frame {start_frame} >= motion end frame {end_frame}"

        sliced_local_rotation = motion.local_rotation[start_frame:end_frame].clone()
        sliced_root_translation = motion.root_translation[start_frame:end_frame].clone()

        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            motion.skeleton_tree,
            sliced_local_rotation,
            sliced_root_translation,
            is_local=True,
        )
        sliced_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=motion.fps)

        return sliced_motion

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
        motion_fpses = []
        (
            motion_files,
            motion_weights,
            motion_timings,
            full_motion_fpses,
            sub_motion_to_motion,
            ref_respawn_offsets,
            motion_labels,
        ) = self._fetch_motion_files(motion_file)

        num_sub_motions = len(sub_motion_to_motion)

        for f in range(num_sub_motions):
            motion_f = sub_motion_to_motion[f]
            curr_file = motion_files[motion_f]
            print(
                "Loading {:d}/{:d} motion files: {:s}".format(
                    f + 1, num_sub_motions, curr_file
                )
            )

            curr_motion = self._load_motion_file(curr_file)

            cur_fps = full_motion_fpses[motion_f]
            if cur_fps is None:
                cur_fps = curr_motion.fps
                
            if cur_fps > target_frame_rate:
                # Not necessary, but we downsample the FPS to save memory
                # do nothing if cur_fps <= target_frame_rate
                curr_motion = self._fix_motion_fps(
                    curr_motion,
                    cur_fps,
                    target_frame_rate,
                    self.skeleton_tree,
                )

            sub_motion = self._slice_motion_file(curr_motion, motion_timings[f])
            motion_fpses.append(float(sub_motion.fps))

            if self.fix_heights:
                sub_motion = self.fix_motion_heights(sub_motion, self.skeleton_tree)

            curr_dt = 1.0 / motion_fpses[f]

            num_frames = sub_motion.global_translation.shape[0]
            curr_len = 1.0 / motion_fpses[f] * (num_frames - 1)

            motion_dt.append(curr_dt)
            motion_num_frames.append(num_frames)

            curr_dof_vels = self._compute_motion_dof_vels(sub_motion)
            sub_motion.dof_vels = curr_dof_vels

            motions.append(sub_motion)
            motion_lengths.append(curr_len)

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
            motion_fps=motion_fpses,
            motion_dt=motion_dt,
            motion_num_frames=motion_num_frames,
            motion_files=tuple(motion_files),
            ref_respawn_offsets=ref_respawn_offsets,
            text_embeddings=text_embeddings,
            has_text_embeddings=has_text_embeddings,
        )

        num_motions = self.num_motions()
        total_len = self.get_total_length()

        print(
            "Loaded {:d} motions with a total length of {:.3f}s.".format(
                num_motions, total_len
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

                    motion_index += 1
        else:
            motion_files = [motion_file]
            motion_weights = [1.0]
            motion_timings = [[0, -1]]
            motion_fpses = [None]
            sub_motion_to_motion = [0]
            ref_respawn_offsets = [0]
            motion_labels = [["", "", ""]]
        return (
            motion_files,
            motion_weights,
            motion_timings,
            motion_fpses,
            sub_motion_to_motion,
            ref_respawn_offsets,
            motion_labels,
        )

    def _calc_frame_blend(self, time, len, num_frames, dt):
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt

        return frame_idx0, frame_idx1, blend

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
        body_ids = self.robot_config.dof_body_ids
        dof_offsets = self.robot_config.dof_offsets

        n = local_rot.shape[0]
        dof_pos = torch.zeros((n, self.robot_config.num_dof), dtype=torch.float, device=self._device)

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

                dof_pos[:, joint_offset: (joint_offset + joint_size)] = formatted_joint
            elif joint_size == 1:
                joint_q = local_rot[:, body_id]
                configured_joint_axis = self.robot_config.joint_axis[j]
                joint_theta, joint_axis = torch_utils.quat_to_angle_axis(
                    joint_q, w_last=True
                )
                if configured_joint_axis == "x":
                    joint_axis = joint_axis[..., 0]
                elif configured_joint_axis == "y":
                    joint_axis = joint_axis[..., 1]
                elif configured_joint_axis == "z":
                    joint_axis[..., 2]

                joint_theta = (
                        joint_theta * joint_axis
                )

                joint_theta = rotations.normalize_angle(joint_theta)
                dof_pos[:, joint_offset] = joint_theta

            else:
                print("Unsupported joint type")
                assert False

        return dof_pos

    def _local_rotation_to_dof_vel(self, local_rot0, local_rot1, dt):
        body_ids = self.robot_config.dof_body_ids
        dof_offsets = self.robot_config.dof_offsets

        dof_vel = torch.zeros([self.robot_config.num_dof], device=self._device)

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
                dof_vel[joint_offset: (joint_offset + joint_size)] = joint_vel

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

    @staticmethod
    def _fix_motion_fps(
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
