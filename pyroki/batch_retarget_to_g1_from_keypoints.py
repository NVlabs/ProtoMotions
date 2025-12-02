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
import time
from typing import Tuple, TypedDict
import glob
import os
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
import yourdfpy

G1_LINK_NAMES = None
N_retarget = 15
N_AUX = 3

# for the local bones alignment cost
direct_pairs = [
    # ("pelvis", "left_shoulder", 1.0),
    # ("pelvis", "right_shoulder", 1.0),
    ("left_shoulder", "left_elbow", 1.0),
    ("right_shoulder", "right_elbow", 1.0),
    ("left_elbow", "left_wrist", 1.0),
    ("right_elbow", "right_wrist", 1.0),
    ("left_hip", "left_knee", 1.0),
    ("right_hip", "right_knee", 1.0),
    ("left_knee", "left_ankle", 1.0),
    ("right_knee", "right_ankle", 1.0),
    ("left_ankle", "left_foot", 1.0),
    ("right_ankle", "right_foot", 1.0),
]


def get_humanoid_retarget_indices() -> jnp.ndarray:
    human_retarget_names = []
    g1_joint_retarget_indices = []

    # NOTE: the order matters here.
    for human_name, g1_name in [
        (
            "pelvis",
            "pelvis_contour_link",
        ),  # source animation key point, target robot link
        ("left_hip", "left_hip_pitch_link"),
        ("right_hip", "right_hip_pitch_link"),
        ("left_knee", "left_knee_link"),
        ("right_knee", "right_knee_link"),
        ("left_ankle", "left_ankle_roll_link"),
        ("right_ankle", "right_ankle_roll_link"),
        ("left_foot", "left_foot_link"),
        ("right_foot", "right_foot_link"),
        ("left_shoulder", "left_shoulder_pitch_link"),
        ("right_shoulder", "right_shoulder_pitch_link"),
        ("left_elbow", "left_elbow_link"),
        ("right_elbow", "right_elbow_link"),
        ("left_wrist", "left_wrist_yaw_link"),
        ("right_wrist", "right_wrist_yaw_link"),
    ]:
        human_retarget_names.append(human_name)
        g1_joint_retarget_indices.append(G1_LINK_NAMES.index(g1_name))

    g1_joint_retarget_indices = jnp.array(g1_joint_retarget_indices)

    return human_retarget_names, g1_joint_retarget_indices


human_retarget_names, g1_joint_retarget_indices = None, None


def load_motion_data(motion_path, source_type, subsample_factor, target_raw_frames):
    """Load and process motion data from a keypoints file.

    Args:
        motion_path: Path to the motion file
        source_type: Source type ('smpl' or 'rigv1')
        subsample_factor: Subsampling factor
        target_raw_frames: Target number of raw frames before subsampling

    Returns:
        Tuple of (simplified_keypoints, keypoint_orientations, left_foot_contact, right_foot_contact, num_timesteps)
    """
    print(f"Loading motion from: {motion_path}")
    motion_data = onp.load(motion_path, allow_pickle=True).item()

    # Compute target subsampled frames from raw frames and subsample factor
    target_subsampled_frames = len(list(range(0, target_raw_frames, subsample_factor)))

    raw_positions = motion_data["positions"]
    raw_orientations = motion_data["orientations"]
    raw_left_foot_contacts = motion_data[
        "left_foot_contacts"
    ]  # [T, 2] - ankle, toebase
    raw_right_foot_contacts = motion_data[
        "right_foot_contacts"
    ]  # [T, 2] - ankle, toebase
    original_raw_frames = raw_positions.shape[0]

    print(f"Original motion length: {original_raw_frames} frames.")

    # Calculate the original number of frames after subsampling for display purposes
    assert original_raw_frames > 0
    original_subsampled_display_count = raw_positions[::subsample_factor].shape[0]
    # Display frames are the minimum of original's useful frames and the buffer's capacity
    num_timesteps = min(original_subsampled_display_count, target_subsampled_frames)

    print(
        f"Motion will be displayed for {num_timesteps} subsampled frames (original subsampled count: {original_subsampled_display_count})."
    )

    # Pad or trim raw data to target_raw_frames for the solver's fixed-size input
    if original_raw_frames >= target_raw_frames:
        processed_positions = raw_positions[:target_raw_frames]
        processed_orientations = raw_orientations[:target_raw_frames]
        processed_left_foot_contacts = raw_left_foot_contacts[:target_raw_frames]
        processed_right_foot_contacts = raw_right_foot_contacts[:target_raw_frames]
    else:
        padding_count = target_raw_frames - original_raw_frames

        last_pos_frame = raw_positions[-1:]
        pos_padding = onp.repeat(last_pos_frame, padding_count, axis=0)
        processed_positions = onp.concatenate((raw_positions, pos_padding), axis=0)

        last_orient_frame = raw_orientations[-1:]
        orient_padding = onp.repeat(last_orient_frame, padding_count, axis=0)
        processed_orientations = onp.concatenate(
            (raw_orientations, orient_padding), axis=0
        )

        last_left_contact_frame = raw_left_foot_contacts[-1:]
        left_contact_padding = onp.repeat(
            last_left_contact_frame, padding_count, axis=0
        )
        processed_left_foot_contacts = onp.concatenate(
            (raw_left_foot_contacts, left_contact_padding), axis=0
        )

        last_right_contact_frame = raw_right_foot_contacts[-1:]
        right_contact_padding = onp.repeat(
            last_right_contact_frame, padding_count, axis=0
        )
        processed_right_foot_contacts = onp.concatenate(
            (raw_right_foot_contacts, right_contact_padding), axis=0
        )

    # Process contact labels BEFORE subsampling for better smoothing
    # Replace OR with average of ankle and toe contacts
    left_foot_contacts_avg = onp.mean(
        processed_left_foot_contacts.astype(float), axis=1
    )[:, None]
    right_foot_contacts_avg = onp.mean(
        processed_right_foot_contacts.astype(float), axis=1
    )[:, None]

    # Apply cross-fade (sliding window average) to smooth contact transitions
    window_size = 5

    def apply_crossfade(contact_flags):
        smoothed = onp.zeros_like(contact_flags)
        for i in range(len(contact_flags)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(contact_flags), i + window_size // 2 + 1)
            smoothed[i] = onp.mean(contact_flags[start_idx:end_idx])
        return smoothed

    left_foot_contacts_smoothed = apply_crossfade(left_foot_contacts_avg)
    right_foot_contacts_smoothed = apply_crossfade(right_foot_contacts_avg)

    # Subsample the processed (padded/trimmed) data for the solver's buffer
    simplified_keypoints = processed_positions[::subsample_factor]

    # Scale keypoints to roughly match the robot's size
    if source_type == "smpl":
        simplified_keypoints_root = simplified_keypoints[:, 0, :]
        simplified_keypoints_local = (
            simplified_keypoints - simplified_keypoints_root[:, None, :]
        )
        simplified_keypoints_lower_body_local = simplified_keypoints_local[:, 1:9, :]
        simplified_keypoints_lower_body_local = (
            simplified_keypoints_lower_body_local
            * onp.array([0.9, 0.9, 0.85])[None, None, :]
        )

        simplified_keypoints_upper_body_local = simplified_keypoints_local[
            :, 9 : N_retarget + N_AUX, :
        ]
        simplified_keypoints_upper_body_local = (
            simplified_keypoints_upper_body_local
            * onp.array([0.9, 0.9, 0.8])[None, None, :]
        )

        simplified_keypoints_local = onp.concatenate(
            [
                simplified_keypoints_lower_body_local,
                simplified_keypoints_upper_body_local,
            ],
            axis=1,
        )

        simplified_keypoints_root = (
            simplified_keypoints_root * onp.array([0.9, 0.9, 0.85])[None, :]
        )
        simplified_keypoints = (
            simplified_keypoints_root[:, None, :] + simplified_keypoints_local
        )
        simplified_keypoints = onp.concatenate(
            [simplified_keypoints_root[:, None, :], simplified_keypoints], axis=1
        )

    elif source_type == "rigv1":
        simplified_keypoints_root = simplified_keypoints[:, 0, :]
        simplified_keypoints_local = (
            simplified_keypoints - simplified_keypoints_root[:, None, :]
        )
        simplified_keypoints_lower_body_local = simplified_keypoints_local[:, 1:9, :]
        simplified_keypoints_lower_body_local = (
            simplified_keypoints_lower_body_local
            * onp.array([0.8, 0.8, 0.75])[None, None, :]
        )

        simplified_keypoints_upper_body_local = simplified_keypoints_local[
            :, 9 : N_retarget + N_AUX, :
        ]
        simplified_keypoints_upper_body_local = (
            simplified_keypoints_upper_body_local
            * onp.array([0.8, 0.8, 0.7])[None, None, :]
        )

        simplified_keypoints_local = onp.concatenate(
            [
                simplified_keypoints_lower_body_local,
                simplified_keypoints_upper_body_local,
            ],
            axis=1,
        )

        simplified_keypoints_root = (
            simplified_keypoints_root * onp.array([0.8, 0.8, 0.75])[None, :]
        )
        simplified_keypoints = (
            simplified_keypoints_root[:, None, :] + simplified_keypoints_local
        )
        simplified_keypoints = onp.concatenate(
            [simplified_keypoints_root[:, None, :], simplified_keypoints], axis=1
        )
    else:
        raise ValueError(f"Invalid source type: {source_type}")

    keypoint_orientations = processed_orientations[::subsample_factor]

    # Subsample the smoothed contact flags
    left_foot_contact = left_foot_contacts_smoothed[::subsample_factor]
    right_foot_contact = right_foot_contacts_smoothed[::subsample_factor]

    expected_pos_shape = (target_subsampled_frames, N_retarget + N_AUX, 3)
    expected_orient_shape = (target_subsampled_frames, N_retarget + N_AUX, 3, 3)
    expected_contact_or_shape = (target_subsampled_frames, 1)
    assert (
        simplified_keypoints.shape == expected_pos_shape
    ), f"Expected positions shape {expected_pos_shape}, got {simplified_keypoints.shape}"
    assert (
        keypoint_orientations.shape == expected_orient_shape
    ), f"Expected orientations shape {expected_orient_shape}, got {keypoint_orientations.shape}"
    assert (
        left_foot_contact.shape == expected_contact_or_shape
    ), f"Expected left foot contacts OR shape {expected_contact_or_shape}, got {left_foot_contact.shape}"
    assert (
        right_foot_contact.shape == expected_contact_or_shape
    ), f"Expected right foot contacts OR shape {expected_contact_or_shape}, got {right_foot_contact.shape}"

    return (
        simplified_keypoints,
        keypoint_orientations,
        left_foot_contact,
        right_foot_contact,
        num_timesteps,
    )


def save_contact_labels(
    output_path, left_foot_contact, right_foot_contact, num_timesteps
):
    """Save processed foot contact labels to disk.

    Args:
        output_path: Path to save the contact labels
        left_foot_contact: Left foot contact array [T, 1]
        right_foot_contact: Right foot contact array [T, 1]
        num_timesteps: Number of actual timesteps (to trim padding)
    """
    # Extract contact labels (already smoothed from load_motion_data), trim to actual length
    left_contacts = left_foot_contact[:num_timesteps].squeeze(-1)  # [K]
    right_contacts = right_foot_contact[:num_timesteps].squeeze(-1)  # [K]

    # Stack into [K, 2] format (raw smoothed values, not binarized)
    foot_contacts = onp.stack([left_contacts, right_contacts], axis=-1)  # [K, 2]

    # Save contact labels
    onp.savez_compressed(output_path, foot_contacts=foot_contacts)
    print(f"Saved contact labels to {output_path} with shape {foot_contacts.shape}")


class RetargetingWeights(TypedDict):
    local_alignment: float
    """Local alignment weight, by matching the relative joint/keypoint positions and angles."""
    global_alignment: float
    """Global alignment weight, by matching the keypoint positions to the robot."""
    root_smoothness: float
    """Root smoothness weight, to penalize the robot's root from jittering too much."""
    joint_smoothness: float
    """Joint smoothness weight, to penalize the robot's joints from jittering too much."""
    self_collision: float
    """Self collision weight, to prevent the robot from colliding with itself."""
    joint_rest_penalty: float
    """Joint rest penalty weight, to penalize certain joints from moving too much."""
    joint_vel_limit: float
    """Joint velocity limit weight, to prevent joint velocities from exceeding limits."""
    foot_contact: float
    """Foot contact weight, to penalize foot movement when in contact."""
    foot_tilt: float
    """Foot tilt weight, to prevent excessive foot tilting when in contact."""


def main():
    """Main function for simplified humanoid retargeting."""
    # Get the directory containing this script for script-relative default paths
    SCRIPT_DIR = Path(__file__).parent.resolve()

    parser = argparse.ArgumentParser(description="Simplified Humanoid Retargeting")
    parser.add_argument(
        "--no-visualize",
        action="store_false",
        dest="visualize",
        help="Run retargeting without visualization and save results to disk.",
    )
    parser.add_argument(
        "--keypoints-folder-path",
        type=str,
        required=True,
        help="Path to the folder containing the keypoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./retargeted_output_motions",
        help="Directory to save retargeted motions in non-visualize mode.",
    )
    parser.add_argument(
        "--urdf-path",
        type=str,
        default=str(
            SCRIPT_DIR / "../protomotions/data/assets/urdf/for_retargeting/g1.urdf"
        ),
        help="Path to the URDF file for the robot. Default is script-relative.",
    )
    parser.add_argument(
        "--mesh-dir",
        type=str,
        default=str(SCRIPT_DIR / "../protomotions/data/assets/mesh/G1"),
        help="Path to the mesh directory for the robot. Default is script-relative.",
    )
    parser.add_argument(
        "--subsample-factor",
        type=int,
        default=1,
        help="Subsample factor for the keypoints. Adjust this to control memory usage and solve speed",
    )
    parser.add_argument(
        "--target-raw-frames",
        type=int,
        default=450,
        help="Target raw frames before subsampling.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip processing motions that already have retargeted output files (useful for resuming interrupted runs).",
    )
    parser.add_argument(
        "--source-type",
        type=str,
        default="smpl",
        help="Source type for the retargeting.",
    )
    parser.add_argument(
        "--save-contacts-only",
        action="store_true",
        help="Skip retargeting and only save processed foot contact labels from source motions.",
    )
    parser.add_argument(
        "--contacts-dir",
        type=str,
        default=None,
        help="Directory to save contact labels. Defaults to {keypoints_folder_path}/contacts",
    )

    args = parser.parse_args()

    # Directory containing motion data files
    keypoints_folder_path = args.keypoints_folder_path

    # Dynamically populate the list of motion data paths
    test_keypoints_paths = sorted(
        glob.glob(os.path.join(keypoints_folder_path, "*.npy"))
    )

    if not test_keypoints_paths:
        print(f"No .npy files found in {keypoints_folder_path}. Exiting.")
        return

    # Subsample factor
    subsample_factor = args.subsample_factor
    TARGET_RAW_FRAMES = args.target_raw_frames  # Target frames before subsampling

    # Early exit for save-contacts-only mode (skip robot/JAX initialization)
    if args.save_contacts_only:
        print(
            "Running in save-contacts-only mode. Extracting foot contact labels from source motions."
        )

        contacts_dir = (
            args.contacts_dir
            if args.contacts_dir
            else os.path.join(args.keypoints_folder_path, "contacts")
        )
        os.makedirs(contacts_dir, exist_ok=True)

        for i, motion_path in enumerate(test_keypoints_paths):
            print(
                f"Processing motion {i+1}/{len(test_keypoints_paths)}: {os.path.basename(motion_path)}"
            )

            # Check if output already exists and skip if requested
            base_filename = os.path.splitext(os.path.basename(motion_path))[0]
            output_filename = f"{base_filename}_contacts.npz"
            output_path = os.path.join(contacts_dir, output_filename)

            if args.skip_existing and os.path.exists(output_path):
                print(f"Output file {output_filename} already exists, skipping...")
                continue

            _, _, left_foot_contact, right_foot_contact, num_timesteps = (
                load_motion_data(
                    motion_path, args.source_type, subsample_factor, TARGET_RAW_FRAMES
                )
            )
            save_contact_labels(
                output_path, left_foot_contact, right_foot_contact, num_timesteps
            )
        return

    # Initialize robot and retargeting infrastructure (only needed for visualization and retargeting)
    global G1_LINK_NAMES

    urdf_path = args.urdf_path
    urdf_mesh_dir = args.mesh_dir
    urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=urdf_mesh_dir)

    robot = pk.Robot.from_urdf(urdf)
    # robot_coll = pk.collision.RobotCollision.from_urdf(urdf)
    robot_coll = None

    if G1_LINK_NAMES is None:
        G1_LINK_NAMES = list(robot.links.names)

    global human_retarget_names, g1_joint_retarget_indices
    human_retarget_names, g1_joint_retarget_indices = get_humanoid_retarget_indices()

    current_motion_index = 0

    # Create connectivity matrix for the G1 joints
    n_retarget = len(g1_joint_retarget_indices)
    g1_retarget_mask = jnp.zeros((n_retarget, n_retarget))
    for link_a, link_b, weight in direct_pairs:
        retarget_idx_a = human_retarget_names.index(link_a)
        retarget_idx_b = human_retarget_names.index(link_b)
        g1_retarget_mask = g1_retarget_mask.at[retarget_idx_a, retarget_idx_b].set(
            weight
        )
        g1_retarget_mask = g1_retarget_mask.at[retarget_idx_b, retarget_idx_a].set(
            weight
        )

    weights_dict = RetargetingWeights(
        local_alignment=1.0,
        global_alignment=4.0,
        root_smoothness=1.0,
        joint_smoothness=4.0,
        self_collision=0.0,  # turned off for now
        joint_rest_penalty=1.0,
        joint_vel_limit=50.0,
        foot_contact=30.0,
        foot_tilt=1.0,
    )

    if args.visualize:
        # Import GUI packages only when visualization is enabled
        import viser
        from viser.extras import ViserUrdf

        # Load initial motion data
        (
            simplified_keypoints,
            keypoint_orientations,
            left_foot_contact,
            right_foot_contact,
            num_timesteps,
        ) = load_motion_data(
            test_keypoints_paths[current_motion_index],
            args.source_type,
            subsample_factor,
            TARGET_RAW_FRAMES,
        )
        server = viser.ViserServer()
        base_frame = server.scene.add_frame("/base", show_axes=False)
        urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
        playing = server.gui.add_checkbox("playing", True)
        # Initialize slider with displayable frames for the first loaded motion
        timestep_slider = server.gui.add_slider(
            "timestep", 0, num_timesteps - 1 if num_timesteps > 0 else 0, 1, 0
        )

        def reset_timeline_callback(_: viser.GuiEvent):
            timestep_slider.value = 0

        reset_timeline_button = server.gui.add_button("Reset Timeline")
        reset_timeline_button.on_click(reset_timeline_callback)

        weights = pk.viewer.WeightTuner(
            server,
            weights_dict,  # type: ignore
        )

        Ts_world_root, joints = None, None

        def generate_trajectory():
            nonlocal Ts_world_root, joints
            gen_button.disabled = True
            retarget_next_button.disabled = True  # Disable while generating
            Ts_world_root, joints = solve_retargeting(
                robot=robot,
                robot_coll=robot_coll,
                target_keypoints=simplified_keypoints,  # Use current motion data
                target_orientations=keypoint_orientations,  # Use current motion data
                left_foot_contact=left_foot_contact,  # Use current motion data
                right_foot_contact=right_foot_contact,  # Use current motion data
                g1_joint_retarget_indices=g1_joint_retarget_indices,
                g1_retarget_mask=g1_retarget_mask,
                weights=weights.get_weights(),  # type: ignore
                subsample_factor=subsample_factor,
            )
            gen_button.disabled = False
            retarget_next_button.disabled = False  # Re-enable after generating

        gen_button = server.gui.add_button("Retarget!")
        gen_button.on_click(lambda _: generate_trajectory())

        def retarget_next_motion(_: viser.GuiEvent):
            nonlocal current_motion_index, Ts_world_root, joints, num_timesteps
            nonlocal \
                simplified_keypoints, \
                keypoint_orientations, \
                left_foot_contact, \
                right_foot_contact
            current_motion_index = (current_motion_index + 1) % len(
                test_keypoints_paths
            )
            (
                simplified_keypoints,
                keypoint_orientations,
                left_foot_contact,
                right_foot_contact,
                num_timesteps,
            ) = load_motion_data(
                test_keypoints_paths[current_motion_index],
                args.source_type,
                subsample_factor,
                TARGET_RAW_FRAMES,
            )

            # Update UI elements that depend on num_timesteps (displayable frames)
            timestep_slider.max = num_timesteps - 1 if num_timesteps > 0 else 0
            timestep_slider.value = 0

            # Clear previous trajectory visualization if any
            Ts_world_root, joints = None, None
            # server.scene.remove("/target_keypoints") # Optional: clear previous keypoints immediately

            generate_trajectory()

        retarget_next_button = server.gui.add_button("Retarget Next")
        retarget_next_button.on_click(retarget_next_motion)

        generate_trajectory()
        assert Ts_world_root is not None and joints is not None

        while True:
            with server.atomic():
                if playing.value and num_timesteps > 0:
                    timestep_slider.value = (timestep_slider.value + 1) % num_timesteps
                tstep = timestep_slider.value

            try:
                base_frame.wxyz = onp.array(Ts_world_root.wxyz_xyz[tstep][:4])
                base_frame.position = onp.array(Ts_world_root.wxyz_xyz[tstep][4:])
                urdf_vis.update_cfg(onp.array(joints[tstep]))

                server.scene.add_point_cloud(
                    "/target_keypoints",
                    onp.array(simplified_keypoints[tstep]),
                    onp.array((0, 0, 255))[None].repeat(
                        simplified_keypoints.shape[1], axis=0
                    ),
                    point_size=0.01,
                )
            except Exception as _:
                pass

            time.sleep(0.03 * subsample_factor)
    else:
        print(
            "Running in non-visualize mode. Retargeting all motions and saving to disk."
        )

        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        for i, motion_path in enumerate(test_keypoints_paths):
            print(
                f"Processing motion {i+1}/{len(test_keypoints_paths)}: {os.path.basename(motion_path)}"
            )

            # Check if output already exists and skip if requested
            base_filename = os.path.splitext(os.path.basename(motion_path))[0]
            output_filename = f"{base_filename}_retargeted.npz"
            output_path = os.path.join(output_dir, output_filename)

            if args.skip_existing and os.path.exists(output_path):
                print(f"Output file {output_filename} already exists, skipping...")
                continue

            (
                simplified_keypoints,
                keypoint_orientations,
                left_foot_contact,
                right_foot_contact,
                num_timesteps,
            ) = load_motion_data(
                motion_path, args.source_type, subsample_factor, TARGET_RAW_FRAMES
            )

            Ts_world_root, joints = solve_retargeting(
                robot=robot,
                robot_coll=robot_coll,
                target_keypoints=simplified_keypoints,
                target_orientations=keypoint_orientations,
                left_foot_contact=left_foot_contact,
                right_foot_contact=right_foot_contact,
                g1_joint_retarget_indices=g1_joint_retarget_indices,
                g1_retarget_mask=g1_retarget_mask,
                weights=weights_dict,
                subsample_factor=subsample_factor,
            )

            # Save results, sliced to the actual motion length
            results_to_save = {
                "base_frame_pos": onp.array(Ts_world_root.wxyz_xyz[:num_timesteps, 4:]),
                "base_frame_wxyz": onp.array(
                    Ts_world_root.wxyz_xyz[:num_timesteps, :4]
                ),
                "joint_angles": onp.array(joints[:num_timesteps]),
            }

            onp.savez_compressed(output_path, **results_to_save)
            print(f"Saved retargeted motion to {output_path}")


@jaxls.Cost.create_factory
def joint_vel_limit_cost(
    var_values: jaxls.VarValues,
    var_joints_curr: jaxls.Var[jnp.ndarray],
    var_joints_prev: jaxls.Var[jnp.ndarray],
    max_vel: float,
    dt: float,
    weight: float,
) -> jax.Array:
    """Joint velocity limit cost to prevent excessive joint velocities."""
    joints_curr = var_values[var_joints_curr]
    joints_prev = var_values[var_joints_prev]

    # Calculate joint velocity
    joint_vel = (joints_curr - joints_prev) / dt

    # Apply penalty when velocity exceeds limit
    excess_vel = jnp.maximum(jnp.abs(joint_vel) - max_vel, 0.0)

    return excess_vel.flatten() * weight


@jaxls.Cost.create_factory
def foot_contact_cost(
    var_values: jaxls.VarValues,
    var_Ts_world_root_curr: jaxls.SE3Var,
    var_Ts_world_root_prev: jaxls.SE3Var,
    var_robot_cfg_curr: jaxls.Var[jnp.ndarray],
    var_robot_cfg_prev: jaxls.Var[jnp.ndarray],
    robot: pk.Robot,
    left_foot_contact: jnp.ndarray,  # [1] - average of ankle and toebase contact (cross-faded)
    right_foot_contact: jnp.ndarray,  # [1] - average of ankle and toebase contact (cross-faded)
    g1_joint_retarget_indices: jnp.ndarray,
    foot_indices: jnp.ndarray,  # [4] - left_ankle_idx, right_ankle_idx, left_foot_idx, right_foot_idx
    weight: float,
) -> jax.Array:
    """When either ankle or toe is in contact, penalize velocity of both ankle and toe,
    and also penalize ankle and toe being at different z heights."""
    T_world_root_curr = var_values[var_Ts_world_root_curr]
    T_world_root_prev = var_values[var_Ts_world_root_prev]
    robot_cfg_curr = var_values[var_robot_cfg_curr]
    robot_cfg_prev = var_values[var_robot_cfg_prev]

    # Get current and previous link positions
    T_root_link_curr = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg_curr))
    T_root_link_prev = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg_prev))
    T_world_link_curr = T_world_root_curr @ T_root_link_curr
    T_world_link_prev = T_world_root_prev @ T_root_link_prev

    # Unpack foot indices
    left_ankle_idx, right_ankle_idx, left_foot_idx, right_foot_idx = foot_indices

    left_ankle_robot_idx = g1_joint_retarget_indices[left_ankle_idx]
    right_ankle_robot_idx = g1_joint_retarget_indices[right_ankle_idx]
    left_foot_robot_idx = g1_joint_retarget_indices[left_foot_idx]
    right_foot_robot_idx = g1_joint_retarget_indices[right_foot_idx]

    # Get current and previous robot foot positions
    robot_positions_curr = T_world_link_curr.translation()
    robot_positions_prev = T_world_link_prev.translation()

    left_ankle_curr = robot_positions_curr[left_ankle_robot_idx]
    right_ankle_curr = robot_positions_curr[right_ankle_robot_idx]
    left_foot_curr = robot_positions_curr[left_foot_robot_idx]
    right_foot_curr = robot_positions_curr[right_foot_robot_idx]

    left_ankle_prev = robot_positions_prev[left_ankle_robot_idx]
    right_ankle_prev = robot_positions_prev[right_ankle_robot_idx]
    left_foot_prev = robot_positions_prev[left_foot_robot_idx]
    right_foot_prev = robot_positions_prev[right_foot_robot_idx]

    # Calculate velocities (position differences between timesteps)
    left_ankle_vel = left_ankle_curr - left_ankle_prev
    right_ankle_vel = right_ankle_curr - right_ankle_prev
    left_foot_vel = left_foot_curr - left_foot_prev
    right_foot_vel = right_foot_curr - right_foot_prev

    # Calculate z-height differences between ankle and toe (current timestep)
    left_ankle_toe_z_diff = left_ankle_curr[2] - left_foot_curr[2]
    right_ankle_toe_z_diff = right_ankle_curr[2] - right_foot_curr[2]

    # Apply contact weighting - when either ankle or toe is in contact, penalize velocities
    left_contact_weight = left_foot_contact[0]  # OR of ankle and toe contact
    right_contact_weight = right_foot_contact[0]  # OR of ankle and toe contact

    # Velocity penalty costs (penalize movement when in contact)
    left_ankle_vel_cost = left_contact_weight * left_ankle_vel
    right_ankle_vel_cost = right_contact_weight * right_ankle_vel
    left_foot_vel_cost = left_contact_weight * left_foot_vel
    right_foot_vel_cost = right_contact_weight * right_foot_vel

    # Z-height consistency costs (ankle and toe should be at similar z when in contact)
    left_z_consistency_cost = left_contact_weight * left_ankle_toe_z_diff
    right_z_consistency_cost = right_contact_weight * right_ankle_toe_z_diff

    return (
        jnp.concatenate(
            [
                left_ankle_vel_cost.flatten(),
                right_ankle_vel_cost.flatten(),
                left_foot_vel_cost.flatten(),
                right_foot_vel_cost.flatten(),
                jnp.array([left_z_consistency_cost]),  # scalar, so wrap in array
                jnp.array([right_z_consistency_cost]),  # scalar, so wrap in array
            ]
        )
        * weight
    )


@jaxls.Cost.create_factory
def foot_tilt_cost(
    var_values: jaxls.VarValues,
    var_Ts_world_root: jaxls.SE3Var,
    var_robot_cfg: jaxls.Var[jnp.ndarray],
    robot: pk.Robot,
    left_foot_contact: jnp.ndarray,  # [1] - average of ankle and toebase contact (cross-faded)
    right_foot_contact: jnp.ndarray,  # [1] - average of ankle and toebase contact (cross-faded)
    g1_joint_retarget_indices: jnp.ndarray,
    foot_indices: jnp.ndarray,  # [4] - left_ankle_idx, right_ankle_idx, left_foot_idx, right_foot_idx
    weight: float,
) -> jax.Array:
    """Cost to penalize foot tilting when in contact - keep z axis up."""
    T_world_root = var_values[var_Ts_world_root]
    robot_cfg = var_values[var_robot_cfg]
    T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
    T_world_link = T_world_root @ T_root_link

    # Unpack foot indices - we need the ankle indices for orientation
    left_ankle_idx, right_ankle_idx, _, _ = foot_indices

    left_ankle_robot_idx = g1_joint_retarget_indices[left_ankle_idx]
    right_ankle_robot_idx = g1_joint_retarget_indices[right_ankle_idx]

    # Get foot orientations (rotation matrices)
    left_foot_ori = T_world_link.rotation().as_matrix()[left_ankle_robot_idx]
    right_foot_ori = T_world_link.rotation().as_matrix()[right_ankle_robot_idx]

    # Penalize tilting: z-axis should point up (rotation_matrix[2,2] should be close to 1)
    left_contact_weight = left_foot_contact[0]  # OR of ankle and toe contact
    right_contact_weight = right_foot_contact[0]  # OR of ankle and toe contact

    left_tilt_residual = left_contact_weight * (left_foot_ori[2, 2] - 1.0)
    right_tilt_residual = right_contact_weight * (right_foot_ori[2, 2] - 1.0)

    return (
        jnp.concatenate(
            [
                jnp.array([left_tilt_residual]),  # scalar, so wrap in array
                jnp.array([right_tilt_residual]),  # scalar, so wrap in array
            ]
        )
        * weight
    )


@jdc.jit
def solve_retargeting(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision | None,
    target_keypoints: jnp.ndarray,
    target_orientations: jnp.ndarray,
    left_foot_contact: jnp.ndarray,
    right_foot_contact: jnp.ndarray,
    g1_joint_retarget_indices: jnp.ndarray,
    g1_retarget_mask: jnp.ndarray,
    weights: RetargetingWeights,
    subsample_factor: int = 1,
) -> Tuple[jaxlie.SE3, jnp.ndarray]:
    """Solve the simplified retargeting problem."""

    n_retarget = len(g1_joint_retarget_indices)
    timesteps = target_keypoints.shape[0]

    # Robot properties.
    # - Joints that should move less for natural humanoid motion.
    joints_to_move_less = jnp.array(
        [
            robot.joints.actuated_names.index(name)
            for name in [
                "waist_roll_joint",
                "right_wrist_pitch_joint",
                "left_wrist_pitch_joint",
            ]
        ]
    )

    # Compute foot indices for contact cost
    foot_indices = jnp.array(
        [
            human_retarget_names.index("left_ankle"),
            human_retarget_names.index("right_ankle"),
            human_retarget_names.index("left_foot"),
            human_retarget_names.index("right_foot"),
        ]
    )

    # Variables.
    class SimplifiedJointsScaleVarG1(
        jaxls.Var[jax.Array], default_factory=lambda: jnp.ones((n_retarget, n_retarget))
    ): ...

    var_joints = robot.joint_var_cls(jnp.arange(timesteps))
    var_Ts_world_root = jaxls.SE3Var(jnp.arange(timesteps))
    var_joints_scale = SimplifiedJointsScaleVarG1(jnp.zeros(timesteps))

    # Initialize root position and orientation using source root at each timestep t
    root_init_se3_list = []
    for t in range(timesteps):
        root_pos_t = target_keypoints[t, 0, :]  # Root position at timestep t
        root_rot_t = target_orientations[t, 0, :, :]  # Root orientation at timestep t

        # Create SE3 transformation for this timestep
        root_se3_t = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_matrix(root_rot_t), root_pos_t
        )
        root_init_se3_list.append(root_se3_t)

    # Stack all timesteps into a single array for batch initialization
    root_init_values = jaxlie.SE3(
        jnp.stack([se3.wxyz_xyz for se3 in root_init_se3_list])
    )

    # Costs.
    costs: list[jaxls.Cost] = []

    # local bones alignment cost
    @jaxls.Cost.create_factory
    def retargeting_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        var_joints_scale: SimplifiedJointsScaleVarG1,
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        """Retargeting factor, with a focus on:
        - matching the relative joint/keypoint positions (vectors).
        - and matching the relative angles between the vectors.
        """
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_root = var_values[var_Ts_world_root]
        T_world_link = T_world_root @ T_root_link

        # Input keypoints are already in the right format
        target_pos = keypoints[:N_retarget, :]  # (N_retarget, 3)
        robot_pos = T_world_link.translation()[jnp.array(g1_joint_retarget_indices)]

        # NxN grid of relative positions.
        delta_target = target_pos[:, None] - target_pos[None, :]
        delta_robot = robot_pos[:, None] - robot_pos[None, :]

        # Vector regularization.
        position_scale = var_values[var_joints_scale][..., None]
        residual_position_delta = (
            (delta_target - delta_robot * position_scale)
            * (1 - jnp.eye(delta_target.shape[0])[..., None])
            * g1_retarget_mask[..., None]
        )

        # Vector angle regularization.
        delta_target_normalized = delta_target / jnp.linalg.norm(
            delta_target + 1e-6, axis=-1, keepdims=True
        )
        delta_robot_normalized = delta_robot / jnp.linalg.norm(
            delta_robot + 1e-6, axis=-1, keepdims=True
        )
        residual_angle_delta = 1 - (
            delta_target_normalized * delta_robot_normalized
        ).sum(axis=-1)
        residual_angle_delta = (
            residual_angle_delta
            * (1 - jnp.eye(residual_angle_delta.shape[0]))
            * g1_retarget_mask
        )

        residual = (
            jnp.concatenate(
                [residual_position_delta.flatten(), residual_angle_delta.flatten()]
            )
            * weights["local_alignment"]
        )
        return residual

    @jaxls.Cost.create_factory
    def scale_regularization(
        var_values: jaxls.VarValues,
        var_joints_scale: SimplifiedJointsScaleVarG1,
    ) -> jax.Array:
        """Regularize the scale of the retargeted joints."""
        # Close to 1.
        res_0 = (var_values[var_joints_scale] - 1.0).flatten() * 1.0
        # Symmetric.
        res_1 = (
            var_values[var_joints_scale] - var_values[var_joints_scale].T
        ).flatten() * 100.0
        # Non-negative.
        res_2 = jnp.clip(-var_values[var_joints_scale], min=0).flatten() * 100.0
        return jnp.concatenate([res_0, res_1, res_2])

    @jaxls.Cost.create_factory
    def pc_alignment_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        var_joints_scale: SimplifiedJointsScaleVarG1,
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        """Soft cost to align the target keypoints to the robot, in the world frame."""
        # position_scale = var_values[var_joints_scale][..., None]

        T_world_root = var_values[var_Ts_world_root]
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_link = T_world_root @ T_root_link
        link_pos = T_world_link.translation()[g1_joint_retarget_indices]

        # hand_aux_link 1
        left_wrist = human_retarget_names.index("left_wrist")
        left_wrist_idx = g1_joint_retarget_indices[left_wrist]
        link_pos_left_wrist = T_world_link.translation()[left_wrist_idx]
        link_rot_mat_left_wrist = T_world_link.rotation().as_matrix()[left_wrist_idx]
        left_hand_aux_pos = link_pos_left_wrist + link_rot_mat_left_wrist @ jnp.array(
            [0.0, 0.0, 0.14]
        )

        right_wrist = human_retarget_names.index("right_wrist")
        right_wrist_idx = g1_joint_retarget_indices[right_wrist]
        link_pos_right_wrist = T_world_link.translation()[right_wrist_idx]
        link_rot_mat_right_wrist = T_world_link.rotation().as_matrix()[right_wrist_idx]
        right_hand_aux_pos = (
            link_pos_right_wrist
            + link_rot_mat_right_wrist @ jnp.array([0.0, 0.0, 0.14])
        )

        # torso aux link
        torso_idx = G1_LINK_NAMES.index("torso_link")
        link_pos_torso = T_world_link.translation()[torso_idx]
        link_rot_mat_torso = T_world_link.rotation().as_matrix()[torso_idx]
        # TODO: this is quite heuristic... the source key points are on pelvis which is lower
        # we use torso here to prevent odd waist rotations
        # but then we need 0.07m height offset...

        torso_aux_pos = link_pos_torso + link_rot_mat_torso @ jnp.array(
            [0.15, 0.0, -0.1]
        )

        link_pos_with_aux = jnp.concatenate(
            [
                link_pos,
                left_hand_aux_pos[None, :],
                right_hand_aux_pos[None, :],
                # left_hand_aux_pos_2[None, :],
                # right_hand_aux_pos_2[None, :],
                torso_aux_pos[None, :],
            ],
            axis=0,
        )  # (N_retarget + N_AUX, 3)

        keypoint_pos = keypoints  # Already in the right format

        # TODO: downweight hand aux a bit?
        keypoint_pos = keypoint_pos.at[-2, :].set(keypoint_pos[-2, :] / 4.0)
        link_pos_with_aux = link_pos_with_aux.at[-2, :].set(
            link_pos_with_aux[-2, :] / 4.0
        )
        keypoint_pos = keypoint_pos.at[-3, :].set(keypoint_pos[-3, :] / 4.0)
        link_pos_with_aux = link_pos_with_aux.at[-3, :].set(
            link_pos_with_aux[-3, :] / 4.0
        )

        # TODO: downweight elbows a bit?
        keypoint_pos = keypoint_pos.at[-6, :].set(keypoint_pos[-6, :] / 4.0)
        link_pos_with_aux = link_pos_with_aux.at[-6, :].set(
            link_pos_with_aux[-6, :] / 4.0
        )
        keypoint_pos = keypoint_pos.at[-7, :].set(keypoint_pos[-7, :] / 4.0)
        link_pos_with_aux = link_pos_with_aux.at[-7, :].set(
            link_pos_with_aux[-7, :] / 4.0
        )

        return (link_pos_with_aux - keypoint_pos).flatten() * weights[
            "global_alignment"
        ]  # (N_retarget + N_AUX) * 3

    @jaxls.Cost.create_factory
    def root_smoothness(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_Ts_world_root_prev: jaxls.SE3Var,
    ) -> jax.Array:
        """Smoothness cost for the robot root pose."""
        return (
            var_values[var_Ts_world_root].inverse() @ var_values[var_Ts_world_root_prev]
        ).log().flatten() * weights["root_smoothness"]

    costs = [
        # Costs that are relatively self-contained to the robot.
        retargeting_cost(
            var_Ts_world_root,
            var_joints,
            var_joints_scale,
            target_keypoints,
        ),
        scale_regularization(var_joints_scale),
        pk.costs.limit_cost(
            jax.tree.map(lambda x: x[None], robot),
            var_joints,
            100.0,
        ),
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            weights["joint_smoothness"],
        ),
        root_smoothness(
            jaxls.SE3Var(jnp.arange(1, timesteps)),
            jaxls.SE3Var(jnp.arange(0, timesteps - 1)),
        ),
        pc_alignment_cost(
            var_Ts_world_root,
            var_joints,
            var_joints_scale,
            target_keypoints,
        ),
        pk.costs.rest_cost(
            var_joints,
            var_joints.default_factory()[None],
            jnp.full(
                var_joints.default_factory().shape, 0.02
            )  # small rest cost for all joints
            .at[joints_to_move_less]
            .set(weights["joint_rest_penalty"])[
                None
            ],  # large rest cost for joints that should move less
        ),
        joint_vel_limit_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            20.0,  # max velocity in rad/s
            subsample_factor / 30.0,  # dt in seconds (accounting for subsampling)
            weights["joint_vel_limit"],
        ),
        # pk.costs.self_collision_cost(
        #     jax.tree.map(lambda x: x[None], robot),
        #     jax.tree.map(lambda x: x[None], robot_coll),
        #     var_joints,
        #     margin=0.01,
        #     weight=weights["self_collision"],
        # ),
    ]

    # Add foot contact costs for each timestep (using v2) - start from t=1 since we need previous timestep
    for t in range(1, timesteps):
        costs.append(
            foot_contact_cost(
                jaxls.SE3Var(t),  # current
                jaxls.SE3Var(t - 1),  # previous
                robot.joint_var_cls(t),  # current cfg
                robot.joint_var_cls(t - 1),  # previous cfg
                robot,
                left_foot_contact[t],
                right_foot_contact[t],
                g1_joint_retarget_indices,
                foot_indices,
                weights["foot_contact"],
            )
        )

    # Add foot tilt costs for each timestep - doesn't need previous timestep
    for t in range(timesteps):
        costs.append(
            foot_tilt_cost(
                jaxls.SE3Var(t),
                robot.joint_var_cls(t),
                robot,
                left_foot_contact[t],
                right_foot_contact[t],
                g1_joint_retarget_indices,
                foot_indices,
                weights["foot_tilt"],  # Reusing foot_tilt weight for tilt cost
            )
        )

    solution = (
        jaxls.LeastSquaresProblem(
            costs, [var_joints, var_Ts_world_root, var_joints_scale]
        )
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make(
                [
                    var_joints,  # Use default initialization for joints
                    var_Ts_world_root.with_value(
                        root_init_values
                    ),  # Use source root initialization
                    var_joints_scale,  # Use default initialization for joint scale
                ]
            ),
            termination=jaxls.TerminationConfig(max_iterations=800),
        )
    )

    return solution[var_Ts_world_root], solution[var_joints]


if __name__ == "__main__":
    main()
