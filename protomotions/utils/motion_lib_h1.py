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
import copy
from typing import Any

import torch
from easydict import EasyDict

from protomotions.utils.motion_lib import MotionLib


class H1_MotionLib(MotionLib):
    def __init__(
        self,
        motion_file,
        dof_body_ids,
        dof_offsets,
        key_body_ids,
        device="cpu",
        ref_height_adjust: float = 0,
        target_frame_rate: int = 30,
        w_last: bool = True,
        create_text_embeddings: bool = False,
        fix_motion_heights: bool = True,
        skeleton_tree: Any = None,
    ):

        super().__init__(
            motion_file=motion_file,
            dof_body_ids=dof_body_ids,
            dof_offsets=dof_offsets,
            key_body_ids=key_body_ids,
            device=device,
            ref_height_adjust=ref_height_adjust,
            target_frame_rate=target_frame_rate,
            w_last=w_last,
            create_text_embeddings=create_text_embeddings,
            fix_motion_heights=fix_motion_heights,
            skeleton_tree=skeleton_tree,
        )

        motions = self.state.motions
        self.register_buffer(
            "dof_pos",
            torch.cat([m.dof_pos for m in motions], dim=0).to(
                device=device, dtype=torch.float32
            ),
            persistent=False,
        )

    @staticmethod
    def _load_motion_file(motion_file):
        motion = EasyDict(torch.load(motion_file))
        return motion

    def _compute_motion_dof_vels(self, motion):
        # We pre-compute the dof vels in h1_humanoid_batch fk.
        return motion.dof_vels

    def fix_motion_heights(self, motion, skeleton_tree):
        body_heights = motion.global_translation[..., 2].clone()
        min_height = body_heights.min()

        motion.global_translation[..., 2] -= min_height
        return motion

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

        sliced_motion = {}

        for key in motion.keys():
            # if is torch.Tensor
            if isinstance(motion[key], torch.Tensor):
                sliced_motion[key] = motion[key][start_frame:end_frame].clone()
            else:
                sliced_motion[key] = copy.deepcopy(motion[key])

        return EasyDict(sliced_motion)
