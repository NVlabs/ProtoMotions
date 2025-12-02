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
import sys
import argparse
import os
import copy


import numpy as np
import torch
import glob
import smplx
import json
import yaml
import math
import shutil
from scipy.spatial.transform import Rotation as sRot
from scipy.spatial.distance import cdist
import trimesh
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from data.smpl.smpl_joint_names import SMPLH_MUJOCO_NAMES
from protomotions.components.scene_lib import (
    Scene,
    BoxSceneObject,
    MeshSceneObject,
    ObjectOptions,
    SceneLib,
)

SMPLX_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
    "right_eye_brow1",
    "right_eye_brow2",
    "right_eye_brow3",
    "right_eye_brow4",
    "right_eye_brow5",
    "left_eye_brow5",
    "left_eye_brow4",
    "left_eye_brow3",
    "left_eye_brow2",
    "left_eye_brow1",
    "nose1",
    "nose2",
    "nose3",
    "nose4",
    "right_nose_2",
    "right_nose_1",
    "nose_middle",
    "left_nose_1",
    "left_nose_2",
    "right_eye1",
    "right_eye2",
    "right_eye3",
    "right_eye4",
    "right_eye5",
    "right_eye6",
    "left_eye4",
    "left_eye3",
    "left_eye2",
    "left_eye1",
    "left_eye6",
    "left_eye5",
    "right_mouth_1",
    "right_mouth_2",
    "right_mouth_3",
    "mouth_top",
    "left_mouth_3",
    "left_mouth_2",
    "left_mouth_1",
    "left_mouth_5",
    "left_mouth_4",
    "mouth_bottom",
    "right_mouth_4",
    "right_mouth_5",
    "right_lip_1",
    "right_lip_2",
    "lip_top",
    "left_lip_2",
    "left_lip_1",
    "left_lip_3",
    "lip_bottom",
    "right_lip_3",
    "right_contour_1",
    "right_contour_2",
    "right_contour_3",
    "right_contour_4",
    "right_contour_5",
    "right_contour_6",
    "right_contour_7",
    "right_contour_8",
    "contour_middle",
    "left_contour_8",
    "left_contour_7",
    "left_contour_6",
    "left_contour_5",
    "left_contour_4",
    "left_contour_3",
    "left_contour_2",
    "left_contour_1",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_sequences(cfg):
    grab_path = cfg.grab_path

    all_seqs = [
        seq
        for seq in glob.glob(grab_path + "/*/*.npz")
        if "verts_body" not in seq and "verts_object" not in seq
    ]

    total_sequences = len(all_seqs)
    if cfg.max_seqs:
        total_sequences = min(total_sequences, cfg.max_seqs)

    if cfg.visualize:
        mv = MeshViewer(offscreen=False)

        # set the camera pose
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = euler([80, -15, 0], "xzx")
        camera_pose[:3, 3] = np.array([-0.5, -4.0, 1.5])
        mv.update_camera_pose(camera_pose)
    else:
        mv = None

    motions_data = []
    scenes_data = []

    sequence_idx = 0
    invalid_seqs = [
        "s5/flashlight_on_2",
        "s5/flashlight_on_1",
        "s1/watch_set_1",
        "s10/pyramidsmall_inspect_1",
        "knob",
    ]

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Processing sequences...", total=total_sequences)

        for sequence in all_seqs:
            if cfg.dataset == "train" and "s10" in sequence:
                continue
            if cfg.dataset == "test" and "s10" not in sequence:
                continue

            # if "apple" not in sequence:
            #     continue
            invalid = False
            for invalid_seq in invalid_seqs:
                if invalid_seq in sequence:
                    invalid = True
            if invalid:
                continue

            print(f"Processing {sequence}")
            motion_info, scene, reasonably_physical = process_sequence(
                cfg, sequence, mv, sequence_idx
            )
            progress.update(task, advance=1)
            if motion_info is None or not reasonably_physical:
                continue

            # Save individual motion and scene configurations
            motion_name = os.path.basename(motion_info["file"]).replace(".motion", "")

            # Create individual motion config
            individual_motion = copy.deepcopy(motion_info)
            individual_motion["idx"] = 0

            # Save individual motion config
            individual_motion_path = os.path.join(
                cfg.out_path, "motions", f"{motion_name}.yaml"
            )
            os.makedirs(os.path.dirname(individual_motion_path), exist_ok=True)
            with open(individual_motion_path, "w") as f:
                yaml.dump({"motions": [individual_motion]}, f)

            # Create individual scene with humanoid_motion_id = 0
            individual_scene = copy.deepcopy(scene)
            individual_scene.humanoid_motion_id = 0

            # Note: Individual scenes are not saved. To use scenes:
            # 1. Create scenes programmatically in your experiment config
            # 2. Or save scenes with SceneLib.save_scenes_to_file(scenes, path)

            motions_data.append(motion_info)
            scenes_data.append(scene)
            sequence_idx += 1

            if cfg.max_seqs and sequence_idx >= cfg.max_seqs:
                break

    if cfg.visualize:
        mv.close_viewer()

    if cfg.out_path:
        # Save motion data
        motions_yaml_path = os.path.join(
            cfg.out_path, "motions", f"{cfg.dataset}_motions.yaml"
        )
        with open(motions_yaml_path, "w") as f:
            yaml.dump({"motions": motions_data}, f)

        # Save scenes using static method (no SceneLib instance needed)
        if len(scenes_data) > 0:
            scenes_lib_path = os.path.join(
                cfg.out_path, "scenes", f"{cfg.dataset}_scenes.pt"
            )
            os.makedirs(os.path.dirname(scenes_lib_path), exist_ok=True)
            SceneLib.save_scenes_to_file(scenes_data, scenes_lib_path)

            print(f"Saved {len(scenes_data)} scenes to {scenes_lib_path}")
            print("To use in experiments:")
            print(
                f"  config = SceneLibConfig(scene_file='{cfg.dataset}_scenes.pt', ...)"
            )
            print("  # SceneLib will be created automatically in BaseEnv")
        else:
            print("No scenes to save")


def process_sequence(cfg, sequence, mv, sequence_idx):
    seq_data = parse_npz(sequence)
    n_comps = seq_data["n_comps"]
    T = seq_data.n_frames
    if "framerate" in seq_data:
        fps = seq_data["framerate"]
    else:
        print("fps not found, defaulting to 120")
        fps = 120  # Default to 120 if not found

    reasonably_physical = True

    sbj_m = smplx.create(
        model_path=cfg.model_path,
        model_type="smplx",
        gender="neutral",
        num_pca_comps=n_comps,
        num_betas=10,
        batch_size=T,
    )

    sbj_mesh = os.path.join(cfg.grab_path, "..", seq_data.body.vtemp)
    sbj_vtemp = np.array(Mesh(filename=sbj_mesh).vertices)
    original_sbj_m = smplx.create(
        model_path=cfg.model_path,
        model_type="smplx",
        gender=seq_data["gender"],
        num_pca_comps=n_comps,
        v_template=sbj_vtemp,
        batch_size=T,
    )

    this_file_path = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(
        this_file_path, "..", "smpl", "smplx_vert_segmentation.json"
    )
    # Load the SMPL-X vertex segmentation data
    with open(json_path, "r") as f:
        smplx_vert_segmentation = json.load(f)

    valid_contacts = np.array(
        smplx_vert_segmentation["rightHand"]
        + smplx_vert_segmentation["rightHandIndex1"]
        + smplx_vert_segmentation["leftHand"]
        + smplx_vert_segmentation["leftHandIndex1"]
    )

    # Update the finger mappings to include all finger joints
    mujoco_finger_names = [
        "L_Thumb1",
        "L_Thumb2",
        "L_Thumb3",
        "L_Index1",
        "L_Index2",
        "L_Index3",
        "L_Middle1",
        "L_Middle2",
        "L_Middle3",
        "L_Ring1",
        "L_Ring2",
        "L_Ring3",
        "L_Pinky1",
        "L_Pinky2",
        "L_Pinky3",
        "R_Thumb1",
        "R_Thumb2",
        "R_Thumb3",
        "R_Index1",
        "R_Index2",
        "R_Index3",
        "R_Middle1",
        "R_Middle2",
        "R_Middle3",
        "R_Ring1",
        "R_Ring2",
        "R_Ring3",
        "R_Pinky1",
        "R_Pinky2",
        "R_Pinky3",
        "L_Wrist",
        "R_Wrist",
    ]

    sbj_params_original = params2torch(seq_data.body.params)
    sbj_params_original["return_full_pose"] = True

    sbj_params = copy.deepcopy(sbj_params_original)
    sbj_params["betas"] = torch.zeros((T, 10), dtype=torch.float32)
    sbj_params["gender"] = "neutral"

    # Process the parameters through the SMPLX model
    smplx_output = sbj_m(**sbj_params)
    original_smplx_output = original_sbj_m(**sbj_params_original)

    # Verify the dimensions
    assert (
        smplx_output.full_pose.shape[1] == 165
    ), f"Expected 165 dimensions, got {smplx_output.full_pose.shape[1]}"

    verts_sbj = to_cpu(smplx_output.vertices)
    original_verts_sbj = to_cpu(original_smplx_output.vertices)
    obj_mesh = os.path.join(cfg.grab_path, "..", seq_data.object.object_mesh)
    obj_mesh = Mesh(filename=obj_mesh)
    obj_vtemp = np.array(obj_mesh.vertices)
    obj_m = ObjectModel(v_template=obj_vtemp, batch_size=T)
    obj_params = params2torch(seq_data.object.params)
    verts_obj = to_cpu(obj_m(**obj_params).vertices)

    original_verts_obj = to_cpu(obj_m(**obj_params).vertices)

    # Realign object based on contact information
    body_contact = seq_data["contact"]["body"]
    obj_contact = seq_data["contact"]["object"]

    # Define hand indices for SMPL-X model
    left_hand_start = smplx_vert_segmentation["leftHand"][0]
    left_hand_end = smplx_vert_segmentation["leftHand"][-1]
    right_hand_start = smplx_vert_segmentation["rightHand"][0]
    right_hand_end = smplx_vert_segmentation["rightHand"][-1]

    left_hand_indices = list(range(left_hand_start, left_hand_end + 1))
    right_hand_indices = list(range(right_hand_start, right_hand_end + 1))
    hand_indices = left_hand_indices + right_hand_indices

    not_hand = np.ones_like(body_contact)
    not_hand[:, hand_indices] = 0

    first_frame_of_contact = None
    in_contact = True
    frames_of_no_contact = []  # List of tuples (start, end)
    current_frames_of_no_contact = None
    body_contact[:, not_hand] = 0

    bodies_in_contact_tensor = torch.zeros(T, len(SMPLH_MUJOCO_NAMES), dtype=torch.bool)
    bodies_in_contact_target_positions = torch.zeros(
        T, len(SMPLH_MUJOCO_NAMES), 4, dtype=torch.float32
    )
    bodies_in_contact_target_positions[..., -1] = -1

    obj_offsets = np.zeros((T, 3))  # Store offsets for each frame

    # Map SMPLX joint indices to MUJOCO joint names
    smplx_to_mujoco_mapping = {
        # Left hand
        "left_thumb1": "L_Thumb1",
        "left_thumb2": "L_Thumb2",
        "left_thumb3": "L_Thumb3",
        "left_index1": "L_Index1",
        "left_index2": "L_Index2",
        "left_index3": "L_Index3",
        "left_middle1": "L_Middle1",
        "left_middle2": "L_Middle2",
        "left_middle3": "L_Middle3",
        "left_ring1": "L_Ring1",
        "left_ring2": "L_Ring2",
        "left_ring3": "L_Ring3",
        "left_pinky1": "L_Pinky1",
        "left_pinky2": "L_Pinky2",
        "left_pinky3": "L_Pinky3",
        # Right hand
        "right_thumb1": "R_Thumb1",
        "right_thumb2": "R_Thumb2",
        "right_thumb3": "R_Thumb3",
        "right_index1": "R_Index1",
        "right_index2": "R_Index2",
        "right_index3": "R_Index3",
        "right_middle1": "R_Middle1",
        "right_middle2": "R_Middle2",
        "right_middle3": "R_Middle3",
        "right_ring1": "R_Ring1",
        "right_ring2": "R_Ring2",
        "right_ring3": "R_Ring3",
        "right_pinky1": "R_Pinky1",
        "right_pinky2": "R_Pinky2",
        "right_pinky3": "R_Pinky3",
        # Add wrists for segmentation
        "left_wrist": "L_Wrist",
        "right_wrist": "R_Wrist",
    }
    mujoco_to_smplx_mapping = {v: k for k, v in smplx_to_mujoco_mapping.items()}

    # After loading the SMPL-X model and before the frame loop, compute vertex-joint mapping once
    def compute_joint_to_vertex_mapping(vertices, joints):
        """Precompute mapping from vertices to closest joints"""
        joint_to_vertices = {joint: [] for joint in mujoco_finger_names}
        joint_positions = (
            joints[0].detach().cpu().numpy()
        )  # Use first frame for mapping

        # For each vertex, find closest joint that has a mujoco mapping
        for vertex_idx in (
            valid_contacts
        ):  # valid_contacts provides a segmented list of hand vertices
            vertex_pos = vertices[0, vertex_idx]  # Use first frame
            distances = np.linalg.norm(joint_positions - vertex_pos, axis=1)
            # Sort joints by distance
            sorted_joint_indices = np.argsort(distances)

            # Find closest joint that has a mujoco mapping
            for joint_idx in sorted_joint_indices:
                smplx_joint_name = SMPLX_JOINT_NAMES[joint_idx]
                if smplx_joint_name in smplx_to_mujoco_mapping:
                    mujoco_joint = smplx_to_mujoco_mapping[smplx_joint_name]
                    joint_to_vertices[mujoco_joint].append(vertex_idx)
                    break

        return joint_to_vertices

    # Compute mapping once
    joint_to_vertices = compute_joint_to_vertex_mapping(
        original_verts_sbj, original_smplx_output.joints
    )

    # Replace the contact detection code with:
    def detect_joint_contacts(joint_position, obj_verts):
        """Detect which joints are in contact with the object based on distance and return closest point"""
        obj_vertices_copy = obj_verts.copy()
        # Calculate distances between each joint and each object vertex
        distances = cdist(joint_position.reshape(1, -1), obj_vertices_copy)
        # Find minimum distance and corresponding vertex index
        closest_vertex_idx = np.argmin(distances, axis=1)
        # Return closest point
        return obj_verts[closest_vertex_idx[0]]

    # In the main processing loop, replace the vertex-based contact detection with:
    for frame in range(T):
        # Get the current object pose
        current_obj_rotation = sRot.from_rotvec(
            to_cpu(obj_params["global_orient"][frame])
        )
        current_obj_translation = to_cpu(obj_params["transl"][frame])

        # X,Y,Z and number of matches for averaging!
        bodies_in_contact_target_positions_this_frame = torch.zeros(
            len(SMPLH_MUJOCO_NAMES), 4, dtype=torch.float32
        )

        for mujoco_joint in joint_to_vertices.keys():
            joint_contact_vertices = joint_to_vertices[mujoco_joint]
            joint_contact_vertices = [
                v for v in joint_contact_vertices if body_contact[frame, v] > 0
            ]
            if len(joint_contact_vertices) == 0:
                continue

            smplx_joint_name = mujoco_to_smplx_mapping[mujoco_joint]
            smplx_joint_idx = SMPLX_JOINT_NAMES.index(smplx_joint_name)
            mujoco_idx = SMPLH_MUJOCO_NAMES.index(mujoco_joint)

            joint_position = original_smplx_output.joints[frame, smplx_joint_idx, :3]
            obj_vertices = verts_obj[frame]
            closest_obj_vertex = detect_joint_contacts(
                joint_position.detach().numpy(), obj_vertices
            )

            bodies_in_contact_target_positions_this_frame[mujoco_idx, :3] = (
                torch.tensor(
                    closest_obj_vertex,
                    dtype=bodies_in_contact_target_positions_this_frame.dtype,
                    device=bodies_in_contact_target_positions_this_frame.device,
                )
            )
            bodies_in_contact_target_positions_this_frame[mujoco_idx, 3] = 1
            bodies_in_contact_tensor[frame, mujoco_idx] = True

        bodies_with_contact = bodies_in_contact_target_positions_this_frame[:, 3] > 0
        for body_idx in range(len(SMPLH_MUJOCO_NAMES)):
            if bodies_with_contact[body_idx]:
                contact_point_local = current_obj_rotation.inv().apply(
                    bodies_in_contact_target_positions_this_frame[body_idx, :3]
                    - current_obj_translation
                )

                bodies_in_contact_target_positions[frame, body_idx, :3] = (
                    torch.from_numpy(contact_point_local)
                )
                bodies_in_contact_target_positions[frame, body_idx, 3] = (
                    0  # Contact is happening now
                )

        body_contact_points = verts_sbj[frame][body_contact[frame] > 0]
        original_body_contact_points = original_verts_sbj[frame][
            body_contact[frame] > 0
        ]
        obj_contact_points = verts_obj[frame][obj_contact[frame] > 0]

        if len(body_contact_points) > 0 and len(obj_contact_points) > 0:
            # In contact!
            if not in_contact:
                current_frames_of_no_contact.append(frame * 1.0 / fps - (1.0 / 3))
                frames_of_no_contact.append(current_frames_of_no_contact)
            in_contact = True
            if first_frame_of_contact is None:
                first_frame_of_contact = frame

            original_body_object_correspondence = cdist(
                original_body_contact_points, obj_contact_points
            )
            closest_obj_indices = np.argmin(original_body_object_correspondence, axis=1)
            closest_obj_points = obj_contact_points[closest_obj_indices]
            offset = (body_contact_points - closest_obj_points).mean(axis=0)
            obj_offsets[frame] = offset
            verts_obj[frame] += obj_offsets[frame]

            # Wait 0.2 seconds after first contact before calculating height offset
            if frame == first_frame_of_contact + int(0.2 * fps):
                verts_obj[
                    first_frame_of_contact : first_frame_of_contact + int(0.2 * fps)
                ] = verts_obj[frame]
                obj_offsets[
                    first_frame_of_contact : first_frame_of_contact + int(0.2 * fps)
                ] = obj_offsets[frame]

            # Check if any contact correspondences are further than 10cm
            new_obj_contact_points = verts_obj[frame][obj_contact[frame] > 0]
            if (
                len(body_contact_points) > 0
                and len(new_obj_contact_points) > 0
                and reasonably_physical
            ):
                closest_obj_points = new_obj_contact_points[closest_obj_indices]
                distances = np.linalg.norm(
                    body_contact_points - closest_obj_points + offset, axis=1
                )
                if np.any(distances > 0.1):  # 10cm = 0.1m
                    reasonably_physical = False
                    print(
                        f"Warning: Found contact correspondences > 10cm apart at frame {frame}"
                    )
                if (
                    np.linalg.norm(obj_offsets[frame] - obj_offsets[frame - 1]) > 0.05
                    and first_frame_of_contact != frame
                ):
                    reasonably_physical = False
                    print(
                        f"Warning: Found object offset jump > 5cm apart at frame {frame}"
                    )
        else:
            if in_contact:
                current_frames_of_no_contact = [frame * 1.0 / fps]
            in_contact = False
            if first_frame_of_contact is not None:
                verts_obj[frame] = verts_obj[frame - 1]
                obj_offsets[frame] = obj_offsets[frame - 1]

    if not in_contact:
        current_frames_of_no_contact.append(
            T * 1.0 / fps
        )  # Add last frame for final range (if ending not in contact)
        frames_of_no_contact.append(current_frames_of_no_contact)

    for frame in range(first_frame_of_contact):
        verts_obj[frame] = verts_obj[first_frame_of_contact]
        obj_offsets[frame] = obj_offsets[first_frame_of_contact]

    assert first_frame_of_contact is not None, "No contact frames found"

    # First, store the original actual contact data - this is a copy of the current tensor
    actual_bodies_in_contact = bodies_in_contact_tensor.clone()

    # Loop through each joint - we'll process each one independently
    num_joints = bodies_in_contact_tensor.shape[1]  # Number of joints
    for joint_idx in range(num_joints):
        # Create a mask for this joint
        joint_contact_mask = actual_bodies_in_contact[:, joint_idx]

        # Find start and end frames of contacts
        contact_frames = torch.where(joint_contact_mask)[0]

        if len(contact_frames) == 0:
            # No contacts for this joint
            continue

        # Fill in small gaps (less than 10 frames)
        contact_ranges = []
        current_range_start = contact_frames[0]
        last_frame = contact_frames[0]

        for frame in contact_frames[1:]:
            # If there's a gap of less than 10 frames, consider it still in contact
            if frame - last_frame < 10:
                # Still in the same contact range
                pass
            else:
                # End of a contact range
                contact_ranges.append((current_range_start, last_frame))
                current_range_start = frame

            last_frame = frame

        # Add the last range
        contact_ranges.append((current_range_start, last_frame))

        # Now fill in the gaps
        for start_frame, end_frame in contact_ranges:
            bodies_in_contact_tensor[start_frame : end_frame + 1, joint_idx] = True

            # Also copy the target positions for filled frames
            for frame in range(start_frame, end_frame + 1):
                if not actual_bodies_in_contact[frame, joint_idx]:
                    # This is a filled frame - find the closest actual contact frame
                    if frame - start_frame < end_frame - frame:
                        # Closer to start
                        reference_frame = start_frame
                    else:
                        # Closer to end
                        reference_frame = end_frame

                    # Copy position from the reference frame
                    bodies_in_contact_target_positions[frame, joint_idx, :3] = (
                        bodies_in_contact_target_positions[
                            reference_frame, joint_idx, :3
                        ]
                    )
                    bodies_in_contact_target_positions[frame, joint_idx, 3] = (
                        0  # Mark as in contact
                    )

    # Smooth the object movement with 1 frames
    for frame in range(1, T):
        verts_obj[frame] = (verts_obj[frame - 1] + verts_obj[frame]) / 2
        obj_offsets[frame] = (obj_offsets[frame - 1] + obj_offsets[frame]) / 2

    # Update obj_params with the new translations
    obj_params["transl"] = torch.tensor(
        to_cpu(obj_params["transl"]) + obj_offsets, dtype=torch.float32
    )

    table_mesh = os.path.join(cfg.grab_path, "..", seq_data.table.table_mesh)
    table_mesh = Mesh(filename=table_mesh)
    table_vtemp = np.array(table_mesh.vertices)
    table_m = ObjectModel(v_template=table_vtemp, batch_size=T)
    table_params = params2torch(seq_data.table.params)

    # Ensure the table is completely flat while preserving yaw rotation
    original_rotation = to_cpu(table_params["global_orient"][0])
    r = sRot.from_rotvec(original_rotation)

    # Get the rotation matrix
    rotation_matrix = r.as_matrix()

    # Extract the roll rotation (rotation around the x-axis)
    roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

    # Create a new rotation matrix that only includes the roll rotation
    flat_rotation_matrix = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )

    # Convert back to rotation vector
    flat_rotvec = sRot.from_matrix(flat_rotation_matrix).as_rotvec()

    # Clip the first axis (roll) to the closest multiple of pi/4
    # Calculate the closest multiple of pi/4
    closest_multiple = round(flat_rotvec[0] / (math.pi / 4)) * (math.pi / 4)

    # Update the first element of flat_rotvec
    flat_rotvec[0] = closest_multiple

    # Update table parameters with the flat rotation
    table_params["global_orient"][:] = torch.tensor(flat_rotvec, dtype=torch.float32)

    # Recalculate table vertices with updated rotation
    verts_table = to_cpu(table_m(**table_params).vertices)

    # Center the table at (0,0) in the x-y plane
    table_center_xy = table_params["transl"][0, :2].clone()
    table_params["transl"][:, :2] -= table_center_xy

    # Adjust object and humanoid positions relative to the table
    obj_params["transl"][:, :2] -= table_center_xy
    sbj_params["transl"][:, :2] -= table_center_xy

    # Get character's min height
    character_minimal_height = verts_sbj[:, :, 2].min().item()

    # Adjust all heights to align with character feet on the floor
    obj_params["transl"][:, 2] -= character_minimal_height

    # Compute new object vertices
    verts_obj = to_cpu(obj_m(**obj_params).vertices)

    # Ensure the table's z-coordinate is at the object's minimum height
    object_minimal_height_first_frame = verts_obj[0, :, 2].min().item()

    # Set the table's z-coordinate to be at the object's minimal height
    table_thickness = 0.05  # The default table thickness is very small
    table_params["transl"][:, 2] = (
        object_minimal_height_first_frame - table_thickness / 2
    )

    sbj_params["transl"][:, 2] -= character_minimal_height

    # Recalculate vertices with updated parameters
    verts_obj = to_cpu(obj_m(**obj_params).vertices)
    verts_table = to_cpu(table_m(**table_params).vertices)
    smplx_output = sbj_m(**sbj_params)
    verts_sbj = to_cpu(smplx_output.vertices)

    # <<< START SUSTAINED FLOATING CHECK >>>
    # Check for sustained periods where the object is floating without hand contact
    # Calculate table surface height (it's constant after adjustment)
    table_surface_z = to_cpu(table_params["transl"][0, 2]) + table_thickness / 2

    # Calculate object's lowest z-coordinate for all frames
    obj_min_z_per_frame = verts_obj[:, :, 2].min(axis=1)

    # Check hand contact for all frames (using the original, non-gap-filled data)
    contact_per_frame = actual_bodies_in_contact.any(axis=1)

    # Define thresholds
    floating_frame_threshold = 10  # Minimum consecutive frames to trigger filter
    floating_height_threshold = 0.10  # Minimum height above table (10cm)

    if reasonably_physical:  # Only check if not already deemed unreasonable
        consecutive_floating_frames = 0
        for frame in range(T):
            in_contact = contact_per_frame[frame]
            object_height_above_table = obj_min_z_per_frame[frame] - table_surface_z
            is_above_threshold = object_height_above_table > floating_height_threshold

            if not in_contact and is_above_threshold:
                # Object is floating without hand contact
                consecutive_floating_frames += 1
            else:
                # Condition not met, reset counter
                consecutive_floating_frames = 0

            # Check if threshold is exceeded
            if consecutive_floating_frames > floating_frame_threshold:
                print(
                    f"Marking sequence {sequence} as not reasonably physical: Object floating for > {floating_frame_threshold} frames."
                )
                reasonably_physical = False  # Set the flag
                break  # No need to check further frames

    # <<< END SUSTAINED FLOATING CHECK >>>

    stl_dir = os.path.join(
        cfg.grab_path,
        "..",
    )
    # Copy PLY file to objects_mesh_dir
    ply_filename = getattr(getattr(seq_data, "object"), "object_mesh")
    ply_src = os.path.join(
        stl_dir,
        "tools/object_meshes/contact_meshes/",
        ply_filename.split("/")[-1],
    )

    if cfg.out_path:
        # Create folder structure under out_path
        motions_dir = os.path.join(cfg.out_path, "motions")
        objects_mesh_dir = os.path.join(cfg.out_path, "objects")

        os.makedirs(motions_dir, exist_ok=True)
        os.makedirs(objects_mesh_dir, exist_ok=True)

        # Save character motion data with full pose
        motion_filename = f"{os.path.basename(os.path.dirname(sequence))}_{os.path.basename(sequence).split('.')[0]}.npz"
        motion_path = os.path.join(motions_dir, motion_filename)
        np.savez(
            motion_path,
            poses=to_cpu(smplx_output.full_pose),
            trans=to_cpu(sbj_params["transl"]),
            betas=to_cpu(sbj_params["betas"][0]),
            gender="neutral",
            mocap_framerate=fps,
        )

        bodies_in_contact_path = os.path.join(
            motions_dir,
            motion_filename.replace(".npz", "_bodies_in_contact.npy"),
        )
        np.save(bodies_in_contact_path, bodies_in_contact_tensor.numpy())

        bodies_in_contact_target_positions_path = os.path.join(
            motions_dir,
            motion_filename.replace(".npz", "_bodies_in_contact_target_positions.npy"),
        )
        np.save(
            bodies_in_contact_target_positions_path,
            bodies_in_contact_target_positions.numpy(),
        )

        # Copy and create URDF for object and table
        ply_dst = os.path.join(objects_mesh_dir, ply_filename.split("/")[-1])
        shutil.copy(ply_src, ply_dst)

        # Calculate table dimensions
        table_vertices = verts_table[0]  # Use the first frame
        min_coords = np.min(table_vertices, axis=0)
        max_coords = np.max(table_vertices, axis=0)
        dimensions = max_coords - min_coords
        dimensions[2] = table_thickness

        # Create object options
        object_options = ObjectOptions(
            density=1000,
            fix_base_link=False,
            angular_damping=0.01,
            linear_damping=0.01,
            max_angular_velocity=100.0,
            vhacd_enabled=True,
            vhacd_params={
                "max_convex_hulls": 32,
                "max_num_vertices_per_ch": 64,
                "resolution": 300000,
            },
        )

        # Create objects for the scene
        scene_objects = []

        # Create mesh object with motion
        object_path = (
            "data/grab/objects/"
            + f"{os.path.splitext(os.path.basename(ply_dst))[0]}.usda"
        )
        object_mesh = MeshSceneObject(
            object_path=object_path,
            options=object_options,
            translation=torch.tensor(to_cpu(obj_params["transl"]), dtype=torch.float32),
            rotation=torch.tensor(
                sRot.from_rotvec(to_cpu(obj_params["global_orient"]).reshape(-1, 3))
                .inv()
                .as_quat()
                .tolist(),
                dtype=torch.float32,
            ),
            fps=fps,
        )
        scene_objects.append(object_mesh)

        # 2. Create the table as a box
        # Calculate table dimensions
        table_vertices = verts_table[0]  # Use the first frame
        min_coords = np.min(table_vertices, axis=0)
        max_coords = np.max(table_vertices, axis=0)
        dimensions = max_coords - min_coords
        dimensions[2] = table_thickness

        table_options = ObjectOptions(
            density=1000,
            fix_base_link=True,
            angular_damping=0.01,
            linear_damping=0.01,
            max_angular_velocity=100.0,
            vhacd_enabled=True,
            vhacd_params={
                "max_convex_hulls": 10,
                "max_num_vertices_per_ch": 64,
                "resolution": 300000,
            },
        )

        table_object = BoxSceneObject(
            width=float(dimensions[0]),
            depth=float(dimensions[1]),
            height=float(dimensions[2]),
            options=table_options,
            translation=to_cpu(table_params["transl"][0]).tolist(),
            rotation=[0.0, 0.0, 0.0, 1.0],
        )
        scene_objects.append(table_object)

        # Create the scene
        scene = Scene(objects=scene_objects, humanoid_motion_id=sequence_idx)

    if cfg.visualize:
        skip_frame = 4

        # Load the PLY mesh
        ply_mesh = trimesh.load(ply_src)

        # Create a Mesh object for the PLY
        ply_vis_mesh = Mesh(
            vertices=ply_mesh.vertices,
            faces=ply_mesh.faces,
            vc=colors["green"],
            smooth=True,
        )

        # Get joint positions
        joints = smplx_output.joints.detach().cpu().numpy()

        for frame in range(0, T, skip_frame):
            # Create meshes with updated vertices
            o_mesh = Mesh(
                vertices=verts_obj[frame], faces=obj_mesh.faces, vc=colors["yellow"]
            )

            for finger in ["R_Pinky3", "R_Ring3", "R_Middle3", "R_Index3", "R_Thumb3"]:
                # Get the finger index in SMPLH_MUJOCO_NAMES
                mujoco_finger_idx = SMPLH_MUJOCO_NAMES.index(finger)

                # Check if this finger is in contact at this frame
                # if not bodies_in_contact_tensor[frame, mujoco_finger_idx]:
                #     print(f"Finger {finger} is not in contact at frame {frame}")
                #     continue

                # Get the target contact position in object local coordinates
                local_contact_pos = bodies_in_contact_target_positions[
                    frame, mujoco_finger_idx, :3
                ]

                # Convert to object's current pose
                obj_pos = obj_params["transl"][frame]
                obj_rot = sRot.from_rotvec(to_cpu(obj_params["global_orient"][frame]))

                # Rotate the local contact point to world coordinates
                world_contact_pos = (
                    obj_rot.apply(local_contact_pos.numpy()) + obj_pos.numpy()
                )

                # Find vertices within 1cm of contact point
                obj_vertices = original_verts_obj[frame]
                distances = np.linalg.norm(obj_vertices - world_contact_pos, axis=1)
                nearby_vertex_indices = np.where(distances < 0.2)[0]

                # Color the nearby vertices
                vertex_mask = np.zeros_like(seq_data["contact"]["object"][frame])
                vertex_mask[nearby_vertex_indices] = 1
                o_mesh.set_vertex_colors(vc=colors["red"], vertex_ids=vertex_mask)

            s_mesh = Mesh(
                vertices=verts_sbj[frame],
                faces=sbj_m.faces,
                vc=colors["pink"],
                smooth=True,
            )
            contact_vertex_ids = np.where(seq_data["contact"]["body"][frame] > 0)[0]
            s_mesh.set_vertex_colors(vc=colors["red"], vertex_ids=contact_vertex_ids)

            # Define colors for each finger
            # finger_colors = {
            #     "Thumb": colors["red"],
            #     "Index": colors["blue"],
            #     "Middle": colors["green"],
            #     "Ring": colors["yellow"],
            #     "Pinky": colors["purple"],
            # }

            # # Color vertices for each finger
            # for hand in ["L_", "R_"]:
            #     for finger, color in finger_colors.items():
            #         finger_vertices = []
            #         # Get vertices for each joint of this finger
            #         for joint_num in ["3"]:
            #             joint_name = f"{hand}{finger}{joint_num}"
            #             finger_vertices.extend(joint_to_vertices[joint_name])
            #         # Color the vertices for this finger
            #         s_mesh.set_vertex_colors(vc=color, vertex_ids=finger_vertices)

            t_mesh = Mesh(
                vertices=verts_table[frame], faces=table_mesh.faces, vc=colors["white"]
            )

            # Apply the current frame's rotation to the PLY mesh
            current_rotation = sRot.from_rotvec(
                to_cpu(obj_params["global_orient"][frame])
            ).inv()
            quaternion = current_rotation.as_quat()
            rotation = sRot.from_quat(quaternion)
            rotated_vertices = rotation.apply(np.array(ply_mesh.vertices))

            from smplx.lbs import batch_rodrigues

            rot_mats = batch_rodrigues(
                obj_params["global_orient"][frame].view(-1, 3)
            ).view([3, 3])

            rotated_vertices = torch.matmul(
                torch.tensor(ply_mesh.vertices, dtype=torch.float32), rot_mats
            )

            # Update the PLY mesh vertices
            ply_vis_mesh.vertices = rotated_vertices
            ply_vis_mesh.vertices += to_cpu(obj_params["transl"][frame])

            # Update the mesh viewer with all meshes
            # mv.set_static_meshes([s_mesh])
            # mv.set_static_meshes([o_mesh, s_mesh, t_mesh, ply_vis_mesh])
            # Draw skeleton
            lhand_mapping = np.array(
                [
                    SMPLX_JOINT_NAMES.index("left_thumb1"),
                    SMPLX_JOINT_NAMES.index("left_thumb2"),
                    SMPLX_JOINT_NAMES.index("left_thumb3"),
                    SMPLX_JOINT_NAMES.index("left_index1"),
                    SMPLX_JOINT_NAMES.index("left_index2"),
                    SMPLX_JOINT_NAMES.index("left_index3"),
                    SMPLX_JOINT_NAMES.index("left_middle1"),
                    SMPLX_JOINT_NAMES.index("left_middle2"),
                    SMPLX_JOINT_NAMES.index("left_middle3"),
                    SMPLX_JOINT_NAMES.index("left_ring1"),
                    SMPLX_JOINT_NAMES.index("left_ring2"),
                    SMPLX_JOINT_NAMES.index("left_ring3"),
                    SMPLX_JOINT_NAMES.index("left_pinky1"),
                    SMPLX_JOINT_NAMES.index("left_pinky2"),
                    SMPLX_JOINT_NAMES.index("left_pinky3"),
                ],
                dtype=np.int32,
            )

            rhand_mapping = np.array(
                [
                    SMPLX_JOINT_NAMES.index("right_thumb1"),
                    SMPLX_JOINT_NAMES.index("right_thumb2"),
                    SMPLX_JOINT_NAMES.index("right_thumb3"),
                    SMPLX_JOINT_NAMES.index("right_index1"),
                    SMPLX_JOINT_NAMES.index("right_index2"),
                    SMPLX_JOINT_NAMES.index("right_index3"),
                    SMPLX_JOINT_NAMES.index("right_middle1"),
                    SMPLX_JOINT_NAMES.index("right_middle2"),
                    SMPLX_JOINT_NAMES.index("right_middle3"),
                    SMPLX_JOINT_NAMES.index("right_ring1"),
                    SMPLX_JOINT_NAMES.index("right_ring2"),
                    SMPLX_JOINT_NAMES.index("right_ring3"),
                    SMPLX_JOINT_NAMES.index("right_pinky1"),
                    SMPLX_JOINT_NAMES.index("right_pinky2"),
                    SMPLX_JOINT_NAMES.index("right_pinky3"),
                ],
                dtype=np.int32,
            )
            lhand_joints = joints[frame][lhand_mapping]
            rhand_joints = joints[frame][rhand_mapping]
            joint_spheres = create_joint_spheres(
                lhand_joints, "blue"
            ) + create_joint_spheres(rhand_joints, "red")
            all_meshes = [o_mesh, t_mesh, s_mesh] + joint_spheres

            for finger in ["R_Pinky3", "R_Ring3", "R_Middle3", "R_Index3", "R_Thumb3"]:
                # Get the finger index in SMPLH_MUJOCO_NAMES
                mujoco_finger_idx = SMPLH_MUJOCO_NAMES.index(finger)

                # Check if this finger is in contact at this frame
                if not bodies_in_contact_tensor[frame, mujoco_finger_idx]:
                    print(f"Finger {finger} is not in contact at frame {frame}")
                    continue

                # Get the target contact position in object local coordinates
                local_contact_pos = bodies_in_contact_target_positions[
                    frame, mujoco_finger_idx, :3
                ]

                # Convert to object's current pose
                obj_pos = obj_params["transl"][frame]
                obj_rot = sRot.from_rotvec(to_cpu(obj_params["global_orient"][frame]))

                # Rotate the local contact point to world coordinates
                world_contact_pos = (
                    obj_rot.apply(local_contact_pos.numpy()) + obj_pos.numpy()
                )

                thumb_spheres = create_joint_spheres(
                    world_contact_pos.reshape(1, 3), "red", radius=0.02
                )
                all_meshes.extend(thumb_spheres)

            mv.set_static_meshes(all_meshes)

    motion_info = {
        "file": os.path.relpath(
            motion_path.replace(".npz", ".motion"),
            os.path.join(cfg.out_path, "motions"),
        ),
        "idx": sequence_idx,
        "time_ranges_of_no_contact": frames_of_no_contact,
        "contacts": f"{os.path.basename(bodies_in_contact_path)}",
        "contact_target_positions": f"{os.path.basename(bodies_in_contact_target_positions_path)}",
        "fps": fps,
        "weight": 1.0,
    }

    return motion_info, scene, reasonably_physical


def create_joint_spheres(joint_positions, color, radius=0.005):
    """Create small spheres for each joint position"""
    import trimesh

    spheres = []
    for pos in joint_positions:
        # Create a small sphere mesh for each joint
        sphere = trimesh.creation.uv_sphere(radius=radius)
        # Move sphere to joint position
        sphere.vertices += pos
        # Convert to Mesh object with red color
        joint_mesh = Mesh(
            vertices=sphere.vertices, faces=sphere.faces, vc=colors[color], smooth=True
        )
        spheres.append(joint_mesh)
    return spheres


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRAB-visualize")

    parser.add_argument(
        "--grab-path",
        required=True,
        type=str,
        help="The path to the downloaded grab data",
    )

    parser.add_argument(
        "--model-path",
        required=True,
        type=str,
        help="The path to the folder containing smplx models",
    )

    parser.add_argument(
        "--out-path",
        type=str,
        help="The path to save the processed data and YAML files",
    )

    parser.add_argument(
        "--visualize", action="store_true", help="Visualize the sequences"
    )

    parser.add_argument(
        "--max-seqs",
        type=int,
        default=None,
        help="The maximum number of sequences to process",
    )

    parser.add_argument(
        "--grab-code-path",
        required=True,
        type=str,
        help="The path to the GRAB code folder containing the 'tools' directory",
    )

    args = parser.parse_args()

    # Add GRAB code path to sys.path
    sys.path.append(args.grab_code_path)

    # Now import the tools
    from tools.objectmodel import ObjectModel
    from tools.meshviewer import Mesh, MeshViewer, colors
    from tools.utils import parse_npz
    from tools.utils import params2torch
    from tools.utils import to_cpu
    from tools.utils import euler
    from tools.cfg_parser import Config

    assert (
        args.out_path is not None or args.visualize
    ), "Please specify either --out-path or --visualize"

    for dataset in ["train", "test"]:
        cfg = {
            "grab_path": args.grab_path,
            "model_path": args.model_path,
            "out_path": args.out_path,
            "visualize": args.visualize,
            "max_seqs": args.max_seqs,
            "dataset": dataset,
        }

        cfg = Config(**cfg)
        process_sequences(cfg)
