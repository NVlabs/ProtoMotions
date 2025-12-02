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
import torch
from protomotions.utils.rotations import quaternion_to_matrix
from typing import List, Dict, Tuple

KEYPOINT_MAPPING_RIGV1 = {
    "pelvis": "Hips",
    "left_hip": "LeftUpLeg",
    "right_hip": "RightUpLeg",
    "left_knee": "LeftLeg",
    "right_knee": "RightLeg",
    "left_ankle": "LeftFoot",
    "right_ankle": "RightFoot",
    "left_foot": "LeftToeBase",
    "right_foot": "RightToeBase",
    "left_shoulder": "LeftArm",
    "right_shoulder": "RightArm",
    "left_elbow": "LeftForeArm",
    "right_elbow": "RightForeArm",
    "left_wrist": "LeftHand",
    "right_wrist": "RightHand",
}

KEYPOINT_MAPPING_SMPL = {
    "pelvis": "Pelvis",
    "left_hip": "L_Hip",
    "right_hip": "R_Hip",
    "left_knee": "L_Knee",
    "right_knee": "R_Knee",
    "left_ankle": "L_Ankle",
    "right_ankle": "R_Ankle",
    "left_foot": "L_Toe",
    "right_foot": "R_Toe",
    "left_shoulder": "L_Shoulder",
    "right_shoulder": "R_Shoulder",
    "left_elbow": "L_Elbow",
    "right_elbow": "R_Elbow",
    "left_wrist": "L_Wrist",
    "right_wrist": "R_Wrist",
}


def get_keypoint_indices(
    kinematic_info, skeleton_format: str = "rigv1"
) -> Tuple[List[str], List[str], List[int]]:
    """
    Get keypoint names and their indices in the MJCF body list.

    Args:
        kinematic_info: Kinematic info object with body_names attribute
        skeleton_format: Either "rigv1" or "smpl" to specify the skeleton format
    """
    if skeleton_format == "rigv1":
        mapping = KEYPOINT_MAPPING_RIGV1
    elif skeleton_format == "smpl":
        mapping = KEYPOINT_MAPPING_SMPL
    else:
        raise ValueError(
            f"Unsupported skeleton format: {skeleton_format}. Must be 'rigv1' or 'smpl'"
        )

    conceptual_keypoint_names = list(mapping.keys())
    mjcf_target_body_names = [mapping[name] for name in conceptual_keypoint_names]

    keypoint_indices_in_mjcf = []
    for mjcf_name in mjcf_target_body_names:
        if mjcf_name in kinematic_info.body_names:
            keypoint_indices_in_mjcf.append(kinematic_info.body_names.index(mjcf_name))
        else:
            raise ValueError(
                f"Keypoint body name '{mjcf_name}' not found in MJCF body names: {kinematic_info.body_names}"
            )

    return conceptual_keypoint_names, mjcf_target_body_names, keypoint_indices_in_mjcf


def extract_keypoints_from_motion_smpl_skel(
    all_body_positions: torch.Tensor,
    all_body_rotations_quat: torch.Tensor,
    keypoint_indices_in_mjcf: List[int],
    conceptual_keypoint_names: List[str],
    device: torch.device,
    flat_feet: bool = True,
    aux_points: bool = True,
    contacts: torch.Tensor = None,
    kinematic_info=None,
) -> Dict[str, torch.Tensor]:
    """
    Extracts keypoints from full-body motion data.
    Optionally applies flat feet correction and adds auxiliary points.
    If contacts and kinematic_info are provided, also extracts foot contact information.

    Args:
        all_body_positions: [T, N_bodies, 3] - body positions
        all_body_rotations_quat: [T, N_bodies, 4] - body rotations as quaternions
        keypoint_indices_in_mjcf: List of indices for keypoints in MJCF body order
        conceptual_keypoint_names: List of conceptual keypoint names
        device: torch device
        flat_feet: Whether to apply flat feet correction
        aux_points: Whether to add auxiliary points
        contacts: Optional [T, N_bodies] - contact labels for all bodies
        kinematic_info: Optional kinematic info object with body_names attribute

    Returns:
        Dictionary containing:
        - 'positions': [T, N_keypoints, 3] - keypoint positions
        - 'orientations': [T, N_keypoints, 3, 3] - keypoint orientations as rotation matrices
        - 'left_foot_contacts': [T, 2] - left foot contact labels (ankle, toebase) if contacts provided
        - 'right_foot_contacts': [T, 2] - right foot contact labels (ankle, toebase) if contacts provided

    Note:
    feet: SMPL +x --> G1/H1 +x
    hands (since G1/H1 zero pose has bent arms instead of T pose):
            SMPL +x --> G1/H1 +z
           SMPL +z --> G1/H1 +y
    """

    # Extract keypoint positions directly
    keypoint_positions = all_body_positions[:, keypoint_indices_in_mjcf, :]

    # Extract keypoint orientations (quaternions) directly and convert to matrices
    keypoint_orientations_quat = all_body_rotations_quat[:, keypoint_indices_in_mjcf, :]
    keypoint_orientations_mat = quaternion_to_matrix(
        keypoint_orientations_quat, w_last=True
    )

    # surgery on the 1st key point
    root_idx = conceptual_keypoint_names.index("pelvis")
    root_offset = torch.tensor(
        [-0.04, 0.0, 0.0], device=device, dtype=keypoint_positions.dtype
    )
    R_root = keypoint_orientations_mat[:, root_idx, :, :]
    keypoint_positions[:, root_idx, :] += torch.einsum(
        "...ij,j->...i", R_root, root_offset
    )

    # surgery on elbow keypoints
    p_local_elbow_mod = torch.tensor(
        [0.0, 0.0, 0.045], device=device, dtype=keypoint_positions.dtype
    )
    left_elbow_idx = conceptual_keypoint_names.index("left_elbow")
    right_elbow_idx = conceptual_keypoint_names.index("right_elbow")
    p_left_elbow = keypoint_positions[:, left_elbow_idx, :]
    R_left_elbow = keypoint_orientations_mat[:, left_elbow_idx, :, :]
    p_left_elbow_new = p_left_elbow + torch.einsum(
        "...ij,j->...i", R_left_elbow, p_local_elbow_mod
    )
    keypoint_positions[:, left_elbow_idx, :] = p_left_elbow_new
    p_right_elbow = keypoint_positions[:, right_elbow_idx, :]
    R_right_elbow = keypoint_orientations_mat[:, right_elbow_idx, :, :]
    p_right_elbow_new = p_right_elbow + torch.einsum(
        "...ij,j->...i", R_right_elbow, p_local_elbow_mod
    )
    keypoint_positions[:, right_elbow_idx, :] = p_right_elbow_new

    if flat_feet:
        # === Custom position calculation for ToeBase keypoints (Flat Feet) ===
        # Instead of using their global positions directly, we calculate them relative to the ankle.
        # This is to use a modified local offset to make feet flat on the ground relative to ankle.

        # Get indices for feet (ToeBase) and ankles (Foot)
        left_ankle_idx = conceptual_keypoint_names.index("left_ankle")
        right_ankle_idx = conceptual_keypoint_names.index("right_ankle")
        left_foot_idx = conceptual_keypoint_names.index("left_foot")
        right_foot_idx = conceptual_keypoint_names.index("right_foot")

        # The modified local position offset for the toe base.
        # Original from MJCF is 0.1193 0.0264 -0.0558, here we use a modified one (flat).
        p_local_mod = torch.tensor(
            [0.15 + 0.03, 0.0, 0.0], device=device, dtype=keypoint_positions.dtype
        )

        # Calculate and update position for RightToeBase ('right_foot')
        p_right_ankle = keypoint_positions[:, right_ankle_idx, :]
        R_right_ankle = keypoint_orientations_mat[:, right_ankle_idx, :, :]
        p_right_toe_base_new = p_right_ankle + torch.einsum(
            "...ij,j->...i", R_right_ankle, p_local_mod
        )
        keypoint_positions[:, right_foot_idx, :] = p_right_toe_base_new

        # Calculate and update position for LeftToeBase ('left_foot')
        # NOTE: The local offset is the same for both feet in this model's coordinate system.
        p_left_ankle = keypoint_positions[:, left_ankle_idx, :]
        R_left_ankle = keypoint_orientations_mat[:, left_ankle_idx, :, :]
        p_left_toe_base_new = p_left_ankle + torch.einsum(
            "...ij,j->...i", R_left_ankle, p_local_mod
        )
        keypoint_positions[:, left_foot_idx, :] = p_left_toe_base_new

        # surgery on ankle keypoint also
        p_local_ankle_mod = torch.tensor(
            [0.03, 0.0, 0.0], device=device, dtype=keypoint_positions.dtype
        )
        p_right_ankle_new = p_right_ankle + torch.einsum(
            "...ij,j->...i", R_right_ankle, p_local_ankle_mod
        )
        p_left_ankle_new = p_left_ankle + torch.einsum(
            "...ij,j->...i", R_left_ankle, p_local_ankle_mod
        )
        keypoint_positions[:, right_ankle_idx, :] = p_right_ankle_new
        keypoint_positions[:, left_ankle_idx, :] = p_left_ankle_new

    if aux_points:
        # === Add auxiliary keypoints for hands and pelvis ===
        left_wrist_idx = conceptual_keypoint_names.index("left_wrist")
        right_wrist_idx = conceptual_keypoint_names.index("right_wrist")

        p_local_hand_aux1 = torch.tensor(
            [0.2, 0.0, 0.0], device=device, dtype=keypoint_positions.dtype
        )
        # p_local_hand_aux2 = torch.tensor([0.0, 0.0, 0.2], device=device, dtype=keypoint_positions.dtype)

        # for p_local_hand in [p_local_hand_aux1, p_local_hand_aux2]:
        for p_local_hand in [p_local_hand_aux1]:
            p_left_wrist = keypoint_positions[:, left_wrist_idx, :]
            R_left_wrist = keypoint_orientations_mat[:, left_wrist_idx, :, :]
            p_left_hand_aux = p_left_wrist + torch.einsum(
                "...ij,j->...i", R_left_wrist, p_local_hand
            )

            p_right_wrist = keypoint_positions[:, right_wrist_idx, :]
            R_right_wrist = keypoint_orientations_mat[:, right_wrist_idx, :, :]
            p_right_hand_aux = p_right_wrist + torch.einsum(
                "...ij,j->...i", R_right_wrist, p_local_hand
            )

            keypoint_positions = torch.cat(
                [
                    keypoint_positions,
                    p_left_hand_aux[:, None, :],
                    p_right_hand_aux[:, None, :],
                ],
                dim=1,
            )
            # NOTE: the orientation of the auxiliary hand is the same as the wrist
            keypoint_orientations_mat = torch.cat(
                [
                    keypoint_orientations_mat,
                    R_left_wrist[:, None, :, :],
                    R_right_wrist[:, None, :, :],
                ],
                dim=1,
            )

        # add pelvis aux, 0.04 ref to surgery on the 1st key point
        pelvis_idx = conceptual_keypoint_names.index("pelvis")
        p_local_pelvis = torch.tensor(
            [0.2 - 0.04, 0.0, 0.0], device=device, dtype=keypoint_positions.dtype
        )
        p_pelvis = keypoint_positions[:, pelvis_idx, :]
        R_pelvis = keypoint_orientations_mat[:, pelvis_idx, :, :]
        p_pelvis_aux = p_pelvis + torch.einsum(
            "...ij,j->...i", R_pelvis, p_local_pelvis
        )
        keypoint_positions = torch.cat(
            [keypoint_positions, p_pelvis_aux[:, None, :]], dim=1
        )
        keypoint_orientations_mat = torch.cat(
            [keypoint_orientations_mat, R_pelvis[:, None, :, :]], dim=1
        )

    result = {
        "positions": keypoint_positions,
        "orientations": keypoint_orientations_mat,
    }

    # Extract foot contact information if contacts and kinematic_info are provided
    if contacts is not None and kinematic_info is not None:
        # Find indices for foot bodies in the motion lib data
        body_name_left_ankle = KEYPOINT_MAPPING_SMPL["left_ankle"]
        body_name_right_ankle = KEYPOINT_MAPPING_SMPL["right_ankle"]
        body_name_left_toebase = KEYPOINT_MAPPING_SMPL["left_foot"]
        body_name_right_toebase = KEYPOINT_MAPPING_SMPL["right_foot"]
        left_ankle_idx = kinematic_info.body_names.index(body_name_left_ankle)  # ankle
        right_ankle_idx = kinematic_info.body_names.index(
            body_name_right_ankle
        )  # ankle
        left_toebase_idx = kinematic_info.body_names.index(
            body_name_left_toebase
        )  # foot/toebase
        right_toebase_idx = kinematic_info.body_names.index(
            body_name_right_toebase
        )  # foot/toebase

        # Extract foot contacts: [T, 2] for each foot (ankle, toebase)
        left_foot_contacts = contacts[:, [left_ankle_idx, left_toebase_idx]]  # [T, 2]
        right_foot_contacts = contacts[
            :, [right_ankle_idx, right_toebase_idx]
        ]  # [T, 2]

        # Convert to binary (0 or 1) format
        left_foot_contacts = left_foot_contacts.float()  # Convert bool to float
        right_foot_contacts = right_foot_contacts.float()  # Convert bool to float

        # Convert to numpy and ensure binary format
        left_foot_contacts_np = (
            left_foot_contacts.cpu().numpy().astype(int)
        )  # [T, 2] - ankle, toebase (0 or 1)
        right_foot_contacts_np = (
            right_foot_contacts.cpu().numpy().astype(int)
        )  # [T, 2] - ankle, toebase (0 or 1)

        # Assert that contact labels are binary (only 0s and 1s)
        assert torch.all(
            torch.isin(torch.from_numpy(left_foot_contacts_np), torch.tensor([0, 1]))
        ), f"Left foot contacts contain non-binary values: {torch.unique(torch.from_numpy(left_foot_contacts_np))}"
        assert torch.all(
            torch.isin(torch.from_numpy(right_foot_contacts_np), torch.tensor([0, 1]))
        ), f"Right foot contacts contain non-binary values: {torch.unique(torch.from_numpy(right_foot_contacts_np))}"

        result["left_foot_contacts"] = left_foot_contacts_np
        result["right_foot_contacts"] = right_foot_contacts_np

    return result


def extract_keypoints_from_motion_rigv1_skel(
    all_body_positions: torch.Tensor,
    all_body_rotations_quat: torch.Tensor,
    keypoint_indices_in_mjcf: List[int],
    conceptual_keypoint_names: List[str],
    device: torch.device,
    flat_feet: bool = True,
    aux_points: bool = True,
    contacts: torch.Tensor = None,
    kinematic_info=None,
) -> Dict[str, torch.Tensor]:
    """
    Extracts keypoints from full-body motion data.
    Optionally applies flat feet correction and adds auxiliary points.
    If contacts and kinematic_info are provided, also extracts foot contact information.

    Args:
        all_body_positions: [T, N_bodies, 3] - body positions
        all_body_rotations_quat: [T, N_bodies, 4] - body rotations as quaternions
        keypoint_indices_in_mjcf: List of indices for keypoints in MJCF body order
        conceptual_keypoint_names: List of conceptual keypoint names
        device: torch device
        flat_feet: Whether to apply flat feet correction
        aux_points: Whether to add auxiliary points
        contacts: Optional [T, N_bodies] - contact labels for all bodies
        kinematic_info: Optional kinematic info object with body_names attribute

    Returns:
        Dictionary containing:
        - 'positions': [T, N_keypoints, 3] - keypoint positions
        - 'orientations': [T, N_keypoints, 3, 3] - keypoint orientations as rotation matrices
        - 'left_foot_contacts': [T, 2] - left foot contact labels (ankle, toebase) if contacts provided
        - 'right_foot_contacts': [T, 2] - right foot contact labels (ankle, toebase) if contacts provided

    Note:
    feet: Rigv1 -y --> G1/H1 +x
    hands (since G1/H1 zero pose has bent arms instead of T pose):
        Rigv1 -y --> G1/H1 +z
        Rigv1 +z --> G1/H1 +y
    """
    # Extract keypoint positions directly
    keypoint_positions = all_body_positions[:, keypoint_indices_in_mjcf, :]

    # Extract keypoint orientations (quaternions) directly and convert to matrices
    keypoint_orientations_quat = all_body_rotations_quat[:, keypoint_indices_in_mjcf, :]
    keypoint_orientations_mat = quaternion_to_matrix(
        keypoint_orientations_quat, w_last=True
    )

    # surgery on the 1st key point
    root_idx = conceptual_keypoint_names.index("pelvis")
    root_offset = torch.tensor(
        [0.0, 0.07, 0.0], device=device, dtype=keypoint_positions.dtype
    )
    R_root = keypoint_orientations_mat[:, root_idx, :, :]
    keypoint_positions[:, root_idx, :] += torch.einsum(
        "...ij,j->...i", R_root, root_offset
    )

    if flat_feet:
        # === Custom position calculation for ToeBase keypoints (Flat Feet) ===
        # Instead of using their global positions directly, we calculate them relative to the ankle.
        # This is to use a modified local offset to make feet flat on the ground relative to ankle.

        # Get indices for feet (ToeBase) and ankles (Foot)
        left_ankle_idx = conceptual_keypoint_names.index("left_ankle")
        right_ankle_idx = conceptual_keypoint_names.index("right_ankle")
        left_foot_idx = conceptual_keypoint_names.index("left_foot")
        right_foot_idx = conceptual_keypoint_names.index("right_foot")

        # The modified local position offset for the toe base.
        # Original from MJCF is [0.0, -0.1607, -0.0585], here we use a modified one (flat).
        p_local_mod = torch.tensor(
            [0.0, -0.1607 - 0.05, 0.0], device=device, dtype=keypoint_positions.dtype
        )

        # Calculate and update position for RightToeBase ('right_foot')
        p_right_ankle = keypoint_positions[:, right_ankle_idx, :]
        R_right_ankle = keypoint_orientations_mat[:, right_ankle_idx, :, :]
        p_right_toe_base_new = p_right_ankle + torch.einsum(
            "...ij,j->...i", R_right_ankle, p_local_mod
        )
        keypoint_positions[:, right_foot_idx, :] = p_right_toe_base_new

        # Calculate and update position for LeftToeBase ('left_foot')
        # NOTE: The local offset is the same for both feet in this model's coordinate system.
        p_left_ankle = keypoint_positions[:, left_ankle_idx, :]
        R_left_ankle = keypoint_orientations_mat[:, left_ankle_idx, :, :]
        p_left_toe_base_new = p_left_ankle + torch.einsum(
            "...ij,j->...i", R_left_ankle, p_local_mod
        )
        keypoint_positions[:, left_foot_idx, :] = p_left_toe_base_new

        # surgery on ankle keypoint also
        p_local_ankle_mod = torch.tensor(
            [0.0, -0.05, 0.0], device=device, dtype=keypoint_positions.dtype
        )
        p_right_ankle_new = p_right_ankle + torch.einsum(
            "...ij,j->...i", R_right_ankle, p_local_ankle_mod
        )
        p_left_ankle_new = p_left_ankle + torch.einsum(
            "...ij,j->...i", R_left_ankle, p_local_ankle_mod
        )
        keypoint_positions[:, right_ankle_idx, :] = p_right_ankle_new
        keypoint_positions[:, left_ankle_idx, :] = p_left_ankle_new

    if aux_points:
        # === Add auxiliary keypoints for hands and pelvis ===
        left_wrist_idx = conceptual_keypoint_names.index("left_wrist")
        right_wrist_idx = conceptual_keypoint_names.index("right_wrist")

        p_local_hand_aux1 = torch.tensor(
            [0.0, -0.20, 0.0], device=device, dtype=keypoint_positions.dtype
        )
        # p_local_hand_aux2 = torch.tensor([0.0, 0.0, 0.20], device=device, dtype=keypoint_positions.dtype)

        # for p_local_hand in [p_local_hand_aux1, p_local_hand_aux2]:
        for p_local_hand in [p_local_hand_aux1]:
            p_left_wrist = keypoint_positions[:, left_wrist_idx, :]
            R_left_wrist = keypoint_orientations_mat[:, left_wrist_idx, :, :]
            p_left_hand_aux = p_left_wrist + torch.einsum(
                "...ij,j->...i", R_left_wrist, p_local_hand
            )

            p_right_wrist = keypoint_positions[:, right_wrist_idx, :]
            R_right_wrist = keypoint_orientations_mat[:, right_wrist_idx, :, :]
            p_right_hand_aux = p_right_wrist + torch.einsum(
                "...ij,j->...i", R_right_wrist, p_local_hand
            )

            keypoint_positions = torch.cat(
                [
                    keypoint_positions,
                    p_left_hand_aux[:, None, :],
                    p_right_hand_aux[:, None, :],
                ],
                dim=1,
            )
            # NOTE: the orientation of the auxiliary hand is the same as the wrist
            keypoint_orientations_mat = torch.cat(
                [
                    keypoint_orientations_mat,
                    R_left_wrist[:, None, :, :],
                    R_right_wrist[:, None, :, :],
                ],
                dim=1,
            )

        # add pelvis aux
        pelvis_idx = conceptual_keypoint_names.index("pelvis")
        p_local_pelvis = torch.tensor(
            [0.0, -0.30, 0.0], device=device, dtype=keypoint_positions.dtype
        )
        p_pelvis = keypoint_positions[:, pelvis_idx, :]
        R_pelvis = keypoint_orientations_mat[:, pelvis_idx, :, :]
        p_pelvis_aux = p_pelvis + torch.einsum(
            "...ij,j->...i", R_pelvis, p_local_pelvis
        )
        keypoint_positions = torch.cat(
            [keypoint_positions, p_pelvis_aux[:, None, :]], dim=1
        )
        keypoint_orientations_mat = torch.cat(
            [keypoint_orientations_mat, R_pelvis[:, None, :, :]], dim=1
        )

    result = {
        "positions": keypoint_positions,
        "orientations": keypoint_orientations_mat,
    }

    # Extract foot contact information if contacts and kinematic_info are provided
    if contacts is not None and kinematic_info is not None:
        # Find indices for foot bodies in the motion lib data
        body_name_left_ankle = KEYPOINT_MAPPING_RIGV1["left_ankle"]
        body_name_right_ankle = KEYPOINT_MAPPING_RIGV1["right_ankle"]
        body_name_left_toebase = KEYPOINT_MAPPING_RIGV1["left_foot"]
        body_name_right_toebase = KEYPOINT_MAPPING_RIGV1["right_foot"]
        left_ankle_idx = kinematic_info.body_names.index(body_name_left_ankle)  # ankle
        right_ankle_idx = kinematic_info.body_names.index(
            body_name_right_ankle
        )  # ankle
        left_toebase_idx = kinematic_info.body_names.index(
            body_name_left_toebase
        )  # foot/toebase
        right_toebase_idx = kinematic_info.body_names.index(
            body_name_right_toebase
        )  # foot/toebase

        # Extract foot contacts: [T, 2] for each foot (ankle, toebase)
        left_foot_contacts = contacts[:, [left_ankle_idx, left_toebase_idx]]  # [T, 2]
        right_foot_contacts = contacts[
            :, [right_ankle_idx, right_toebase_idx]
        ]  # [T, 2]

        # Convert to binary (0 or 1) format
        left_foot_contacts = left_foot_contacts.float()  # Convert bool to float
        right_foot_contacts = right_foot_contacts.float()  # Convert bool to float

        # Convert to numpy and ensure binary format
        left_foot_contacts_np = (
            left_foot_contacts.cpu().numpy().astype(int)
        )  # [T, 2] - ankle, toebase (0 or 1)
        right_foot_contacts_np = (
            right_foot_contacts.cpu().numpy().astype(int)
        )  # [T, 2] - ankle, toebase (0 or 1)

        # Assert that contact labels are binary (only 0s and 1s)
        assert torch.all(
            torch.isin(torch.from_numpy(left_foot_contacts_np), torch.tensor([0, 1]))
        ), f"Left foot contacts contain non-binary values: {torch.unique(torch.from_numpy(left_foot_contacts_np))}"
        assert torch.all(
            torch.isin(torch.from_numpy(right_foot_contacts_np), torch.tensor([0, 1]))
        ), f"Right foot contacts contain non-binary values: {torch.unique(torch.from_numpy(right_foot_contacts_np))}"

        result["left_foot_contacts"] = left_foot_contacts_np
        result["right_foot_contacts"] = right_foot_contacts_np

    return result


def extract_keypoints_from_motion(
    all_body_positions: torch.Tensor,
    all_body_rotations_quat: torch.Tensor,
    keypoint_indices_in_mjcf: List[int],
    conceptual_keypoint_names: List[str],
    device: torch.device,
    skeleton_format: str = "rigv1",
    flat_feet: bool = True,
    aux_points: bool = True,
    contacts: torch.Tensor = None,
    kinematic_info=None,
) -> Dict[str, torch.Tensor]:
    """
    Generic function to extract keypoints from motion data.
    Dispatches to the appropriate skeleton-specific function.

    Args:
        all_body_positions: [T, N_bodies, 3] - body positions
        all_body_rotations_quat: [T, N_bodies, 4] - body rotations as quaternions
        keypoint_indices_in_mjcf: List of indices for keypoints in MJCF body order
        conceptual_keypoint_names: List of conceptual keypoint names
        device: torch device
        skeleton_format: Either "rigv1" or "smpl" to specify the skeleton format
        flat_feet: Whether to apply flat feet correction
        aux_points: Whether to add auxiliary points
        contacts: Optional [T, N_bodies] - contact labels for all bodies
        kinematic_info: Optional kinematic info object with body_names attribute

    Returns:
        Dictionary containing:
        - 'positions': [T, N_keypoints, 3] - keypoint positions
        - 'orientations': [T, N_keypoints, 3, 3] - keypoint orientations as rotation matrices
        - 'left_foot_contacts': [T, 2] - left foot contact labels (ankle, toebase) if contacts provided
        - 'right_foot_contacts': [T, 2] - right foot contact labels (ankle, toebase) if contacts provided
    """
    if skeleton_format == "rigv1":
        return extract_keypoints_from_motion_rigv1_skel(
            all_body_positions=all_body_positions,
            all_body_rotations_quat=all_body_rotations_quat,
            keypoint_indices_in_mjcf=keypoint_indices_in_mjcf,
            conceptual_keypoint_names=conceptual_keypoint_names,
            device=device,
            flat_feet=flat_feet,
            aux_points=aux_points,
            contacts=contacts,
            kinematic_info=kinematic_info,
        )
    elif skeleton_format == "smpl":
        return extract_keypoints_from_motion_smpl_skel(
            all_body_positions=all_body_positions,
            all_body_rotations_quat=all_body_rotations_quat,
            keypoint_indices_in_mjcf=keypoint_indices_in_mjcf,
            conceptual_keypoint_names=conceptual_keypoint_names,
            device=device,
            flat_feet=flat_feet,
            aux_points=aux_points,
            contacts=contacts,
            kinematic_info=kinematic_info,
        )
    else:
        raise ValueError(
            f"Unsupported skeleton format: {skeleton_format}. Must be 'rigv1' or 'smpl'"
        )


def get_mjcf_path(skeleton_format: str) -> str:
    """
    Get the appropriate MJCF file path for the given skeleton format.

    Args:
        skeleton_format: Either "rigv1" or "smpl"

    Returns:
        Path to the appropriate MJCF file
    """
    if skeleton_format == "rigv1":
        return "protomotions/data/assets/mjcf/rigv1_humanoid.xml"
    elif skeleton_format == "smpl":
        return "protomotions/data/assets/mjcf/smpl_humanoid.xml"  # Update this path as needed
    else:
        raise ValueError(
            f"Unsupported skeleton format: {skeleton_format}. Must be 'rigv1' or 'smpl'"
        )
