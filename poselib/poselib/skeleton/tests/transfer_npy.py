# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
from ...core import Tensor, SO3, Quaternion, Vector3D
from ..skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion

tpose = np.load(
    "/home/serfcx/DL_Animation/rl_mimic/data/skeletons/flex_tpose.npy"
).item()

local_rotation = SO3.from_numpy(tpose["local_rotation"], dtype="float32")
root_translation = Vector3D.from_numpy(tpose["root_translation"], dtype="float32")
skeleton_tree = tpose["skeleton_tree"]
parent_indices = Tensor.from_numpy(skeleton_tree["parent_indices"], dtype="int32")
local_translation = Vector3D.from_numpy(
    skeleton_tree["local_translation"], dtype="float32"
)
node_names = skeleton_tree["node_names"]
skeleton_tree = SkeletonTree(node_names, parent_indices, local_translation)
skeleton_state = SkeletonState.from_rotation_and_root_translation(
    skeleton_tree=skeleton_tree, r=local_rotation, t=root_translation, is_local=True
)

skeleton_state.to_file(
    "/home/serfcx/DL_Animation/rl_mimic/data/skeletons/flex_tpose_new.npy"
)
