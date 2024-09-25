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

import torch
from torch import Tensor

import numpy as np
import os.path
import yaml

from phys_anim.envs.env_utils.general import StepTracker
from phys_anim.envs.masked_mimic.masked_mimic_utils import build_sparse_target_poses

from smpl_sim.smpllib.smpl_joint_names import (
    SMPL_MUJOCO_NAMES,
    SMPL_BONE_ORDER_NAMES,
    SMPLH_BONE_ORDER_NAMES,
    SMPLH_MUJOCO_NAMES,
)
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot

import tempfile
from scipy.spatial.transform import Rotation as sRot
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState, SkeletonTree

from isaac_utils import rotations
from isaac_utils import torch_utils

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from phys_anim.envs.masked_mimic.tasks.inbetweening.isaacgym import (
        MaskedMimicInbetweeningHumanoid,
    )
else:
    MaskedMimicInbetweeningHumanoid = object


class BaseMaskedMimicInbetweening(MaskedMimicInbetweeningHumanoid):  # type: ignore[misc]
    def __init__(self, config, device):
        self.text_command = None
        self.start_pose = None
        self.end_pose = None

        super().__init__(config, device)

        self.transition_time = 0
        self.hold_time = 0.2
        self.fsm_state = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    ###############################################################
    # Environment step logic
    ###############################################################
    def compute_observations(self, env_ids=None):
        ep_time = self.progress_buf * self.dt

        fsm_state_1 = ep_time < self.hold_time
        fsm_state_2 = ep_time >= self.hold_time

        time_to_target = -ep_time.clone()
        time_to_target[fsm_state_1] += self.hold_time
        time_to_target[fsm_state_2] += self.transition_time

        self.fsm_state[:] = fsm_state_1.int() + 2 * fsm_state_2.int()

        time_to_target = time_to_target.clamp(min=0.1)

        self.target_pose_time[:] = self.motion_times + time_to_target

        self.mask_everything()
        super().compute_observations(env_ids)
        self.mask_everything()

        if self.text_command is not None:
            self.motion_text_embeddings[:] = self.text_command

    def mask_everything(self):
        # By Default mask everything out. Individual tasks will override this.
        self.masked_mimic_target_poses_masks[:] = False
        self.masked_mimic_target_poses_masks[:, -2] = True
        self.masked_mimic_target_poses_masks[:, -1] = True
        self.masked_mimic_target_bodies_masks[:] = False
        self.target_pose_obs_mask[:] = True
        self.object_bounding_box_obs_mask[:] = False
        self.motion_text_embeddings_mask[:] = self.text_command is not None

    def update_inference_parameters(self):
        # Get the appropriate temporary directory
        tmp_dir = tempfile.gettempdir()
        inference_file_path = os.path.join(
            tmp_dir, "masked_mimic_inference_parameters.yaml"
        )

        # Print the directory and inference file path
        print(f"Temporary directory: {tmp_dir}")
        print(f"Inference file path: {inference_file_path}")

        if not os.path.exists(inference_file_path):
            tmp_parameters = {
                "constrained_joints": [
                    {"body_name": "L_Ankle", "constraint_state": 1},
                    {"body_name": "R_Ankle", "constraint_state": 1},
                    {"body_name": "Pelvis", "constraint_state": 1},
                    {"body_name": "Head", "constraint_state": 1},
                    {"body_name": "L_Hand", "constraint_state": 1},
                    {"body_name": "R_Hand", "constraint_state": 1},
                ],
                "text_conditioned": False,
                "text_command": "a person stands up and walks",
                "start_pose": {
                    "file": None,
                    "offset_from_origin": [0, 0, 0],
                    "rotation": [0, 0, 0],
                },
                "end_pose": {
                    "file": None,
                    "offset_from_origin": [0, 0, 0],
                    "rotation": [0, 0, 0],
                },
                "transition_time": 3,
            }
            with open("/tmp/masked_mimic_inference_parameters.yaml", "w") as f:
                yaml.dump(tmp_parameters, f)

        with open(inference_file_path, "r") as f:
            inference_parameters = yaml.load(f, Loader=yaml.SafeLoader)

        try:
            if inference_parameters["text_command"] is None:
                self.text_command = None
            else:
                from transformers import AutoTokenizer, XCLIPTextModel

                model = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32")
                tokenizer = AutoTokenizer.from_pretrained(
                    "microsoft/xclip-base-patch32"
                )

                text_command = [inference_parameters["text_command"]]
                with torch.inference_mode():
                    inputs = tokenizer(
                        text_command, padding=True, truncation=True, return_tensors="pt"
                    )
                    outputs = model(**inputs)
                    pooled_output = outputs.pooler_output  # pooled (EOS token) states
                    self.text_command = pooled_output[0].to(self.device)
                    self.motion_text_embeddings[:] = self.text_command

            if inference_parameters["start_pose"]["file"] is not None:
                global_translation, global_rotation, dof_pos = parse_file_to_pose(
                    filename=inference_parameters["start_pose"]["file"],
                    motion_lib=self.motion_lib,
                    config=self.config,
                )
                global_translation, global_rotation = apply_offset_and_rotation_to_pose(
                    translation=global_translation,
                    rotation=global_rotation,
                    target_translation=inference_parameters["start_pose"][
                        "offset_from_origin"
                    ],
                    target_rotation=inference_parameters["start_pose"]["rotation"],
                )
                self.start_pose = {
                    "translation": global_translation,
                    "rotation": global_rotation,
                    "dof_pos": dof_pos,
                }
            else:
                self.start_pose = None
                print("!!! Please provide a start pose !!!")

            if inference_parameters["end_pose"]["file"] is not None:
                global_translation, global_rotation, dof_pos = parse_file_to_pose(
                    filename=inference_parameters["end_pose"]["file"],
                    motion_lib=self.motion_lib,
                    config=self.config,
                )
                global_translation, global_rotation = apply_offset_and_rotation_to_pose(
                    translation=global_translation,
                    rotation=global_rotation,
                    target_translation=inference_parameters["end_pose"][
                        "offset_from_origin"
                    ],
                    target_rotation=inference_parameters["end_pose"]["rotation"],
                )
                self.end_pose = {
                    "translation": global_translation,
                    "rotation": global_rotation,
                    "dof_pos": dof_pos,
                }
            else:
                self.end_pose = None
                print("!!! Please provide a end pose !!!")

            self.transition_time = (
                self.hold_time + inference_parameters["transition_time"]
            )

            self.config.masked_mimic_masking.joint_masking.masked_mimic_fixed_conditioning = inference_parameters[
                "constrained_joints"
            ]
            self.config.masked_mimic_masking.joint_masking.masked_mimic_time_gap_probability = (
                1.0
            )
            self.config.masked_mimic_masking.joint_masking.masked_mimic_repeat_mask_probability = (
                1.0
            )
            self.config.masked_mimic_masking.joint_masking.time_gap_mask_min_steps = (
                1000
            )
            self.config.masked_mimic_masking.joint_masking.time_gap_mask_max_steps = (
                1001
            )
            self.config.masked_mimic_masking.joint_masking.with_conditioning_max_gap_probability = (
                1.0
            )
            self.config.masked_mimic_masking.object_bounding_box_visible_prob = 0.0
            self.config.masked_mimic_masking.motion_text_embeddings_visible_prob = (
                1.0 if inference_parameters["text_conditioned"] else 0.0
            )
            self.config.masked_mimic_masking.target_pose_visible_prob = 1.0

            self.time_gap_mask_steps = StepTracker(
                self.config.num_envs,
                min_steps=self.config.masked_mimic_masking.joint_masking.time_gap_mask_min_steps,
                max_steps=self.config.masked_mimic_masking.joint_masking.time_gap_mask_max_steps,
                device=self.device,
            )
            self.long_term_gap_probs = (
                torch.ones(self.config.num_envs, dtype=torch.float, device=self.device)
                * self.config.masked_mimic_masking.joint_masking.with_conditioning_max_gap_probability
            )
            self.visible_object_bounding_box_probs = (
                torch.ones(self.config.num_envs, dtype=torch.float, device=self.device)
                * self.config.masked_mimic_masking.joint_masking.object_bounding_box_visible_prob
            )
            self.visible_text_embeddings_probs = (
                torch.ones(self.config.num_envs, dtype=torch.float, device=self.device)
                * self.config.masked_mimic_masking.motion_text_embeddings_visible_prob
            )
            self.visible_target_pose_probs = (
                torch.ones(self.config.num_envs, dtype=torch.float, device=self.device)
                * self.config.masked_mimic_masking.target_pose_visible_prob
            )

            self.config.fixed_motion_id = 0

            # Let's force it to reset!
            self.motion_times[:] = 1000
            self.reset_track_steps.reset_steps()
            self.progress_buf[:] = 0
        except Exception as e:
            print("Error processing inference parameters.")
            print("An easy fix will be to delete the yaml file and try again.")
            print(e)

    def reset_actors(self, env_ids):
        super().reset_actors(env_ids)

        # Make sure in user-control mode that the history isn't visible.
        self.valid_hist_buf.set_all(False)

    def reset_track(self, env_ids, new_motion_ids=None):
        super().reset_track(env_ids, new_motion_ids)

        # Make sure in user-control mode that the history isn't visible.
        self.valid_hist_buf.set_all(False)

    def build_sparse_target_poses(self, raw_future_times):
        """
        override the default func to provide the inbetweening target poses
        """
        num_future_steps = raw_future_times.shape[1]

        motion_ids = self.motion_ids.unsqueeze(-1).tile([1, num_future_steps])
        flat_ids = motion_ids.view(-1)

        raw_future_times[:, -2] = raw_future_times[:, -1] - self.dt
        flat_times = raw_future_times.view(-1)

        ref_state = self.motion_lib.get_mimic_motion_state(flat_ids, flat_times)
        flat_target_pos, flat_target_rot, flat_target_vel = (
            ref_state.rb_pos,
            ref_state.rb_rot,
            ref_state.rb_vel,
        )

        current_state = self.get_bodies_state()
        cur_gt, cur_gr = current_state.body_pos, current_state.body_rot
        # First remove the height based on the current terrain, then remove the offset to get back to the ground-truth data position
        cur_gt[:, :, -1:] -= self.get_ground_heights(cur_gt[:, 0, :2]).view(
            self.num_envs, 1, 1
        )

        target_pos = flat_target_pos.reshape(self.num_envs, num_future_steps, -1, 3)
        target_rot = flat_target_rot.reshape(self.num_envs, num_future_steps, -1, 4)

        first_pose = self.fsm_state == 1
        if torch.any(first_pose) and self.start_pose is not None:
            target_pos[first_pose] = self.start_pose["translation"].view(1, 1, -1, 3)
            target_rot[first_pose] = self.start_pose["rotation"].view(1, 1, -1, 4)
        second_pose = self.fsm_state == 2
        if torch.any(second_pose) and self.end_pose is not None:
            target_pos[second_pose] = self.end_pose["translation"].view(1, 1, -1, 3)
            target_rot[second_pose] = self.end_pose["rotation"].view(1, 1, -1, 4)

        flat_target_pos = target_pos.reshape(*flat_target_pos.shape)
        flat_target_rot = target_rot.reshape(*flat_target_rot.shape)

        return build_sparse_target_poses(
            cur_gt=cur_gt,
            cur_gr=cur_gr,
            flat_target_pos=flat_target_pos,
            flat_target_rot=flat_target_rot,
            flat_target_vel=flat_target_vel,
            masked_mimic_conditionable_bodies_ids=self.masked_mimic_conditionable_bodies_ids,
            num_future_steps=num_future_steps,
            num_envs=self.num_envs,
            w_last=self.w_last,
        )


def parse_file_to_pose(filename, motion_lib, config):
    assert (
        "smpl" in config.robot.asset.robot_type
    ), "Humanoid type must be one of smpl, smplx, smplh"
    if config.robot.asset.robot_type in [
        "smpl_humanoid",
    ]:
        mujoco_joint_names = SMPL_MUJOCO_NAMES
        joint_names = SMPL_BONE_ORDER_NAMES
        humanoid_type = "smpl"
    elif config.robot.asset.robot_type in ["smplx_box_humanoid"]:
        mujoco_joint_names = SMPLH_MUJOCO_NAMES
        joint_names = SMPLH_BONE_ORDER_NAMES
        humanoid_type = "smplx"
    else:
        raise NotImplementedError(
            f"Not supported asset {config.robot.asset.robot_type}."
        )

    robot_cfg = {
        "mesh": False,
        "rel_joint_lm": False,
        "upright_start": True,
        "remove_toe": False,
        "real_weight": True,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True,
        "replace_feet": True,
        "masterfoot": False,
        "big_ankle": True,
        "freeze_hand": False,
        "box_body": True,
        "master_range": 30,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
        "model": humanoid_type,
        "sim": "isaacgym",
    }

    smpl_local_robot = LocalRobot(
        robot_cfg,
        data_dir="smpllib/data/smpl",
    )

    motion_data = np.load(filename, allow_pickle=True)

    amass_pose = motion_data["pose"]
    amass_trans = motion_data["trans"]

    pose_aa = torch.tensor(amass_pose).view(-1, 25, 3)[:, 1:].view(-1, 24 * 3)
    amass_trans = torch.tensor(amass_trans).view(-1, 25, 3)[:, 1].view(-1, 3)

    gender = "neutral"
    gender_number = [0]

    num_repeat = 100

    pose_aa = pose_aa.repeat(num_repeat, 1)
    amass_trans = amass_trans.repeat(num_repeat, 1)

    motion_data = {
        "pose_aa": pose_aa.numpy(),
        "trans": amass_trans.numpy(),
        "gender": gender,
    }

    smpl_2_mujoco = [
        joint_names.index(q) for q in mujoco_joint_names if q in joint_names
    ]
    batch_size = num_repeat
    # batch_size = pose_aa.shape[0]

    if humanoid_type == "smpl":
        pose_aa = np.concatenate(
            [motion_data["pose_aa"][:, :66], np.zeros((batch_size, 6))], axis=1
        )  # TODO: need to extract correct handle rotations instead of zero
        # pose_quat = sRot.from_rotvec(pose_aa.reshape(-1, 3)).as_quat().reshape(batch_size, 24, 4)[..., smpl_2_mujoco, :]
        pose_quat = (
            sRot.from_euler("xyz", pose_aa.reshape(-1, 3), degrees=True)
            .as_quat()
            .reshape(batch_size, 24, 4)[..., smpl_2_mujoco, :]
        )
    else:  # smplx or smplh
        pose_aa = np.concatenate(
            [motion_data["pose_aa"][:, :66], motion_data["pose_aa"][:, 75:]], axis=-1
        )
        pose_quat = (
            sRot.from_rotvec(pose_aa.reshape(-1, 3))
            .as_quat()
            .reshape(batch_size, 52, 4)[..., smpl_2_mujoco, :]
        )

    # betas = None will default as all zeros
    tmp_dir = tempfile.gettempdir()
    smpl_local_robot.load_from_skeleton(
        betas=None, gender=gender_number, objs_info=None
    )
    smpl_local_robot.write_xml(f"{tmp_dir}/smpl/smpl_humanoid_1.xml")
    skeleton_tree = SkeletonTree.from_mjcf(f"{tmp_dir}/smpl/smpl_humanoid_1.xml")

    root_trans_offset = torch.from_numpy(
        motion_data["trans"] + skeleton_tree.local_translation[0].numpy()
    )

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here.
        torch.from_numpy(pose_quat),
        root_trans_offset,
        is_local=True,
    )
    sk_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=1)

    # Upright start
    B = pose_aa.shape[0]
    pose_quat_global = (
        (
            sRot.from_quat(sk_motion.global_rotation.reshape(-1, 4).numpy())
            * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
        )
        .as_quat()
        .reshape(B, -1, 4)
    )

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_quat_global),
        root_trans_offset,
        is_local=False,
    )

    global_translation = new_sk_state.global_translation
    global_rotation = new_sk_state.global_rotation
    local_rot = new_sk_state.local_rotation
    dof_pos = motion_lib._local_rotation_to_dof(local_rot, "exp_map")

    return (
        global_translation[0].float().to(motion_lib.device),
        global_rotation[0].float().to(motion_lib.device),
        dof_pos[0].float().to(motion_lib.device),
    )


@torch.jit.script_if_tracing
def apply_offset_and_rotation_to_pose(
    translation: Tensor,
    rotation: Tensor,
    target_translation: List[float],
    target_rotation: float,
):
    root_translation = translation[0].clone()
    translation[:] -= root_translation.view(1, -1)

    root_rotation = rotation[0, :]
    inv_heading = torch_utils.calc_heading_quat_inv(
        root_rotation.view(1, -1), w_last=True
    )
    inv_heading_expand = inv_heading.repeat((translation.shape[0], 1))

    local_translation = rotations.quat_rotate(
        inv_heading_expand, translation, w_last=True
    )
    local_rotation = rotations.quat_mul(inv_heading_expand, rotation, w_last=True)

    target_rotation = torch.tensor(
        target_rotation, device=rotation.device, dtype=torch.float
    ).view(1, 3)
    target_rotation_quat = rotations.quat_from_euler_xyz(
        roll=target_rotation[:, 0],
        pitch=target_rotation[:, 1],
        yaw=target_rotation[:, 2],
        w_last=True,
    ).view(1, 4)
    target_rotation_quat_expand = target_rotation_quat.expand(translation.shape[0], -1)
    target_translation = torch.tensor(
        target_translation, dtype=torch.float, device=rotation.device
    ).view(1, 3)

    translation = (
        rotations.quat_rotate(
            target_rotation_quat_expand, local_translation, w_last=True
        )
        + target_translation
    )
    rotation = rotations.quat_mul(
        target_rotation_quat_expand, local_rotation, w_last=True
    )

    return translation, rotation
