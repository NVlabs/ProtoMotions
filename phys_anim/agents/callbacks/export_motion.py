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

from phys_anim.agents.callbacks.base_callback import RL_EvalCallback
from phys_anim.agents.ppo import PPO
from phys_anim.envs.humanoid.common import BaseHumanoid

import torch
from pathlib import Path
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
import os.path as osp
import shutil

import numpy as np
import yaml
import pickle
from scipy.spatial.transform import Rotation as sRot


class ExportMotion(RL_EvalCallback):
    training_loop: PPO
    env: BaseHumanoid

    def __init__(self, config, training_loop: PPO):
        super().__init__(config, training_loop)
        self.record_dir = Path(config.record_dir)
        self.record_dir.mkdir(exist_ok=True, parents=True)

    def on_pre_evaluate_policy(self):
        # Doing this in two lines because of type annotation issues.
        env: BaseHumanoid = self.training_loop.env
        self.env = env

    def on_post_evaluate_policy(self):
        self.write_recordings()

    def write_recordings(self):
        fps = np.round(1.0 / self.env.dt)
        for idx in range(self.env.num_envs):
            trajectory_data = self.env.motion_recording

            save_dir = self.record_dir / f"{(idx + self.config.index_offset):03d}"
            save_dir.mkdir(exist_ok=True, parents=True)

            if self.config.store_poselib:
                skeleton_tree = self.env.motion_lib.state.motions[0].skeleton_tree

                curr_root_pos = torch.stack(
                    [root_pos[idx] for root_pos in trajectory_data["root_pos"]]
                )
                curr_root_pos[..., :2] = self.env.convert_to_global_coords(
                    curr_root_pos[..., :2],
                    self.env.env_offsets[idx, :2].view(1, 1, 2).cpu(),
                )
                curr_body_rot = torch.stack(
                    [global_rot[idx] for global_rot in trajectory_data["global_rot"]]
                )

                sk_state = SkeletonState.from_rotation_and_root_translation(
                    skeleton_tree, curr_body_rot, curr_root_pos, is_local=False
                )
                sk_motion = SkeletonMotion.from_skeleton_state(sk_state, fps=fps)

                sk_motion.to_file(str(save_dir / f"trajectory_poselib_{idx}.npy"))

                if "target_poses" in trajectory_data:
                    target_poses = torch.tensor(
                        np.stack(
                            [
                                target_pose[idx]
                                for target_pose in trajectory_data["target_poses"]
                            ]
                        )
                    )
                    if not hasattr(self.env, "export_motion_dont_convert_to_global"):
                        target_poses[..., :2] = self.env.convert_to_global_coords(
                            target_poses[..., :2],
                            self.env.env_offsets[idx, :2].view(1, 1, 2).cpu(),
                        )
                    np.save(
                        str(save_dir / f"target_poses_{idx}.npy"),
                        target_poses.cpu().numpy(),
                    )

                if hasattr(self.env, "object_ids") and self.env.object_ids[idx] >= 0:
                    object_id = self.env.object_ids[idx].item()
                    object_category, object_name = self.env.spawned_object_names[
                        object_id
                    ].split("_")
                    object_offset = self.env.object_offsets[object_category]
                    object_pos = self.env.scene_position[object_id].clone()
                    object_pos[0] += object_offset[0]
                    object_pos[1] += object_offset[1]

                    object_bbs = self.env.object_id_to_object_bounding_box[
                        object_id
                    ].clone()

                    # Add the height offset for the bounding box to match in global coords
                    object_center_xy = self.env.object_root_states[object_id, :2].view(
                        1, 2
                    )
                    terrain_height_below_object = self.env.get_ground_heights(
                        object_center_xy
                    ).view(1)
                    object_bbs[:, -1] += terrain_height_below_object

                    object_info = {
                        "object_pos": [
                            object_pos[0].item(),
                            object_pos[1].item(),
                            object_pos[2].item(),
                        ],
                        "object_name": object_name,
                        "object_bbox": object_bbs.cpu().tolist(),
                    }
                    with open(str(save_dir / f"object_info_{idx}.yaml"), "w") as file:
                        yaml.dump(object_info, file)
                    category_root = osp.join(
                        self.env.config.object_asset_root, object_category
                    )
                    # copy urdf and obj files to new dir, using copy functions
                    shutil.copyfile(
                        str(osp.join(category_root, f"{object_name}.urdf")),
                        str(save_dir / f"{object_name}.urdf"),
                    )
                    shutil.copyfile(
                        str(osp.join(category_root, f"{object_name}.obj")),
                        str(save_dir / f"{object_name}.obj"),
                    )

            else:
                if "smpl" in self.env.config.robot.asset.robot_type:
                    from smpl_sim.smpllib.smpl_joint_names import (
                        SMPL_MUJOCO_NAMES,
                        SMPL_BONE_ORDER_NAMES,
                        SMPLH_BONE_ORDER_NAMES,
                        SMPLH_MUJOCO_NAMES,
                    )

                    if self.env.config.robot.asset.robot_type in [
                        "smpl_humanoid",
                    ]:
                        mujoco_joint_names = SMPL_MUJOCO_NAMES
                        joint_names = SMPL_BONE_ORDER_NAMES
                    elif self.env.config.robot.asset.robot_type in [
                        "smplx_box_humanoid"
                    ]:
                        mujoco_joint_names = SMPLH_MUJOCO_NAMES
                        joint_names = SMPLH_BONE_ORDER_NAMES
                    else:
                        raise NotImplementedError

                    mujoco_2_smpl = [
                        mujoco_joint_names.index(q)
                        for q in joint_names
                        if q in mujoco_joint_names
                    ]
                else:
                    raise NotImplementedError

                pre_rot = sRot.from_quat([0.5, 0.5, 0.5, 0.5])

                body_quat = torch.stack(trajectory_data["rigid_body_rot"])[:, idx]
                root_trans = torch.stack(trajectory_data["rigid_body_pos"])[
                    :, idx, 0, :
                ]

                N = body_quat.shape[0]

                skeleton_tree = self.env.motion_lib.state.motions[0].skeleton_tree

                # offset = skeleton_tree.local_translation[0]
                offset = root_trans[0].clone()
                offset[2] = 0
                root_trans_offset = root_trans - offset

                pose_quat = (
                    (sRot.from_quat(body_quat.reshape(-1, 4).numpy()) * pre_rot)
                    .as_quat()
                    .reshape(N, -1, 4)
                )
                new_sk_state = SkeletonState.from_rotation_and_root_translation(
                    skeleton_tree,
                    torch.from_numpy(pose_quat),
                    root_trans.cpu(),
                    is_local=False,
                )
                local_rot = new_sk_state.local_rotation
                pose_aa = (
                    sRot.from_quat(local_rot.reshape(-1, 4).numpy())
                    .as_rotvec()
                    .reshape(N, -1, 3)
                )
                pose_aa = pose_aa[:, mujoco_2_smpl, :].reshape(1, N, -1)

                with open(save_dir / f"trajectory_pose_aa_{idx}.pkl", "wb") as f:
                    pickle.dump(
                        {
                            "pose": pose_aa,
                            "trans": root_trans_offset.unsqueeze(0).cpu().numpy(),
                            "shape": np.zeros((N, 10)),
                            "gender": "neutral",
                        },
                        f,
                    )

        for key in self.env.motion_recording.keys():
            self.env.motion_recording[key] = []
