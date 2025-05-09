# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from ...core import *
from ..skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion

import numpy as np
import torch

from ...visualization.common import (
    plot_skeleton_state,
    plot_skeleton_motion_interactive,
)

from ...visualization.plt_plotter import Matplotlib3DPlotter
from ...visualization.skeleton_plotter_tasks import (
    Draw3DSkeletonMotion,
    Draw3DSkeletonState,
)


def test_skel_tree():
    skel_tree = SkeletonTree.from_mjcf(
        "/home/serfcx/DL_Animation/rl_mimic/data/skeletons/humanoid_mimic_mod_2_noind.xml",
        backend="pytorch",
    )
    skel_tree_rec = SkeletonTree.from_dict(skel_tree.to_dict(), backend="pytorch")
    # assert skel_tree.to_str() == skel_tree_rec.to_str()
    print(skel_tree.node_names)
    print(skel_tree.local_translation)
    print(skel_tree.parent_indices)
    skel_state = SkeletonState.zero_pose(skeleton_tree=skel_tree)
    plot_skeleton_state(task_name="draw_skeleton", skeleton_state=skel_state)
    skel_state = skel_state.drop_nodes_by_names(["right_hip", "left_hip"])
    plot_skeleton_state(task_name="draw_skeleton", skeleton_state=skel_state)


def test_skel_motion():
    skel_motion = SkeletonMotion.from_file(
        "/tmp/tmp.npy", backend="pytorch", load_context=True
    )

    plot_skeleton_motion_interactive(skel_motion)


def test_grad():
    source_motion = SkeletonMotion.from_file(
        "c:\\Users\\bmatusch\\carbmimic\\data\\motions\\JogFlatTerrain_01_ase.npy",
        backend="pytorch",
        device="cuda:0",
    )
    source_tpose = SkeletonState.from_file(
        "c:\\Users\\bmatusch\\carbmimic\\data\\skeletons\\fox_tpose.npy",
        backend="pytorch",
        device="cuda:0",
    )

    target_tpose = SkeletonState.from_file(
        "c:\\Users\\bmatusch\\carbmimic\\data\\skeletons\\flex_tpose.npy",
        backend="pytorch",
        device="cuda:0",
    )
    target_skeleton_tree = target_tpose.skeleton_tree

    joint_mapping = {
        "upArm_r": "right_shoulder",
        "upArm_l": "left_shoulder",
        "loArm_r": "right_elbow",
        "loArm_l": "left_elbow",
        "upLeg_r": "right_hip",
        "upLeg_l": "left_hip",
        "loLeg_r": "right_knee",
        "loLeg_l": "left_knee",
        "foot_r": "right_ankle",
        "foot_l": "left_ankle",
        "hips": "pelvis",
        "neckA": "neck",
        "spineA": "abdomen",
    }

    rotation_to_target_skeleton = quat_from_angle_axis(
        angle=torch.tensor(90.0).float(),
        axis=torch.tensor([1, 0, 0]).float(),
        degree=True,
    )

    target_motion = source_motion.retarget_to(
        joint_mapping=joint_mapping,
        source_tpose_local_rotation=source_tpose.local_rotation,
        source_tpose_root_translation=source_tpose.root_translation,
        target_skeleton_tree=target_skeleton_tree,
        target_tpose_local_rotation=target_tpose.local_rotation,
        target_tpose_root_translation=target_tpose.root_translation,
        rotation_to_target_skeleton=rotation_to_target_skeleton,
        scale_to_target_skeleton=0.01,
    )

    target_state = SkeletonState(
        target_motion.tensor[800, :],
        target_motion.skeleton_tree,
        target_motion.is_local,
    )

    skeleton_tree = target_state.skeleton_tree
    root_translation = target_state.root_translation
    global_translation = target_state.global_translation

    q = np.zeros((len(skeleton_tree), 4), dtype=np.float32)
    q[..., 3] = 1.0
    q = torch.from_numpy(q)
    max_its = 10000

    task = Draw3DSkeletonState(task_name="", skeleton_state=target_state)
    plotter = Matplotlib3DPlotter(task)

    for i in range(max_its):
        r = quat_normalize(q)
        s = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree, r=r, t=root_translation, is_local=True
        )
        print("  quat norm: {}".format(q.norm(p=2, dim=-1).mean().numpy()))

        task.update(s)
        plotter.update()
    plotter.show()


test_grad()