import copy
from typing import Any

import numpy as np
import torch
from easydict import EasyDict
from protomotions.envs.mimic.mimic_utils import dof_to_local

from protomotions.simulator.base_simulator.config import RobotConfig
from protomotions.utils.motion_lib import MotionLib


class H1_MotionLib(MotionLib):
    def __init__(
            self,
            motion_file,
            robot_config: RobotConfig,
            key_body_ids,
            device="cpu",
            ref_height_adjust: float = 0,
            target_frame_rate: int = 30,
            w_last: bool = True,
            create_text_embeddings: bool = False,
            local_rot_conversion: torch.Tensor = None,
            fix_motion_heights: bool = True,
            skeleton_tree: Any = None,
    ):

        super().__init__(
            motion_file=motion_file,
            robot_config=robot_config,
            key_body_ids=key_body_ids,
            device=device,
            ref_height_adjust=ref_height_adjust,
            target_frame_rate=target_frame_rate,
            w_last=w_last,
            create_text_embeddings=create_text_embeddings,
            local_rot_conversion=local_rot_conversion,
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

    def _load_motion_file(self, motion_file):
        motion = EasyDict(torch.load(motion_file))
        motion.local_rotation = dof_to_local(motion.dof_pos, self.robot_config.dof_offsets,
                                             self.robot_config.joint_axis, True)

        return motion

    def _compute_motion_dof_vels(self, motion):
        # We pre-compute the dof vels in fk.
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

    @staticmethod
    def _fix_motion_fps(motion, orig_fps, target_frame_rate, skeleton_tree):
        skip = int(np.round(orig_fps / target_frame_rate))

        downsampled_motion = {}
        for key in motion.keys():
            # if is torch.Tensor
            if isinstance(motion[key], torch.Tensor):
                downsampled_motion[key] = motion[key][::skip].clone()
            else:
                downsampled_motion[key] = copy.deepcopy(motion[key])

        downsampled_motion["fps"] = target_frame_rate
        return EasyDict(downsampled_motion)
