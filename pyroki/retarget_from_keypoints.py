# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
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

from __future__ import annotations

import argparse
import functools
import glob
import importlib
import os
import time
from pathlib import Path
from typing import Any, Tuple, TypedDict

import numpy as onp
import yaml

N_RETARGET = 15
N_AUX = 3

# For the local bones alignment cost.
DIRECT_PAIRS = [
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


class RetargetingWeights(TypedDict):
    local_alignment: float
    global_alignment: float
    root_smoothness: float
    joint_smoothness: float
    self_collision: float
    joint_rest_penalty: float
    joint_vel_limit: float
    foot_contact: float
    foot_tilt: float


def _resolve_config_path(config_dir: Path, config_value: str) -> str:
    path = Path(config_value)
    if path.is_absolute():
        return str(path)
    return str((config_dir / path).resolve())


def load_robot_config(config_path: str) -> dict[str, Any]:
    resolved_path = Path(config_path).resolve()
    with resolved_path.open("r", encoding="utf-8") as config_file:
        robot_config = yaml.safe_load(config_file)

    if not isinstance(robot_config, dict):
        raise ValueError(f"Robot config at {resolved_path} must deserialize to a dict")

    robot_config = dict(robot_config)
    robot_config["urdf_path"] = _resolve_config_path(
        resolved_path.parent, robot_config["urdf_path"]
    )
    robot_config["mesh_dir"] = _resolve_config_path(
        resolved_path.parent, robot_config["mesh_dir"]
    )
    return robot_config


def get_humanoid_retarget_indices(
    link_names: list[str], robot_config: dict[str, Any]
) -> Tuple[list[str], onp.ndarray]:
    human_retarget_names: list[str] = []
    joint_retarget_indices: list[int] = []

    for human_name, robot_link_name in robot_config["keypoint_mapping"]:
        human_retarget_names.append(human_name)
        joint_retarget_indices.append(link_names.index(robot_link_name))

    return human_retarget_names, onp.array(joint_retarget_indices, dtype=onp.int32)


def build_retarget_mask(human_retarget_names: list[str]) -> onp.ndarray:
    n_retarget = len(human_retarget_names)
    retarget_mask = onp.zeros((n_retarget, n_retarget), dtype=onp.float32)

    for link_a, link_b, weight in DIRECT_PAIRS:
        retarget_idx_a = human_retarget_names.index(link_a)
        retarget_idx_b = human_retarget_names.index(link_b)
        retarget_mask[retarget_idx_a, retarget_idx_b] = weight
        retarget_mask[retarget_idx_b, retarget_idx_a] = weight

    return retarget_mask


def _apply_source_scaling(
    simplified_keypoints: onp.ndarray,
    source_type: str,
    scale_factors: dict[str, Any],
) -> onp.ndarray:
    if source_type not in scale_factors:
        raise ValueError(f"Invalid source type: {source_type}")

    source_scale = scale_factors[source_type]

    simplified_keypoints_root = simplified_keypoints[:, 0, :]
    simplified_keypoints_local = simplified_keypoints - simplified_keypoints_root[:, None, :]

    simplified_keypoints_lower_body_local = simplified_keypoints_local[:, 1:9, :]
    simplified_keypoints_lower_body_local = (
        simplified_keypoints_lower_body_local
        * onp.array(source_scale["lower_body"])[None, None, :]
    )

    simplified_keypoints_upper_body_local = simplified_keypoints_local[
        :, 9 : N_RETARGET + N_AUX, :
    ]
    simplified_keypoints_upper_body_local = (
        simplified_keypoints_upper_body_local
        * onp.array(source_scale["upper_body"])[None, None, :]
    )

    simplified_keypoints_local = onp.concatenate(
        [
            simplified_keypoints_lower_body_local,
            simplified_keypoints_upper_body_local,
        ],
        axis=1,
    )

    simplified_keypoints_root = (
        simplified_keypoints_root * onp.array(source_scale["root"])[None, :]
    )
    simplified_keypoints = (
        simplified_keypoints_root[:, None, :] + simplified_keypoints_local
    )
    return onp.concatenate(
        [simplified_keypoints_root[:, None, :], simplified_keypoints], axis=1
    )


def load_motion_data(
    motion_path: str,
    source_type: str,
    subsample_factor: int,
    target_raw_frames: int,
    scale_factors: dict[str, Any],
):
    """Load and process motion data from a keypoints file."""
    print(f"Loading motion from: {motion_path}")
    motion_data = onp.load(motion_path, allow_pickle=True).item()

    target_subsampled_frames = len(list(range(0, target_raw_frames, subsample_factor)))

    raw_positions = motion_data["positions"]
    raw_orientations = motion_data["orientations"]
    raw_left_foot_contacts = motion_data["left_foot_contacts"]
    raw_right_foot_contacts = motion_data["right_foot_contacts"]
    original_raw_frames = raw_positions.shape[0]

    print(f"Original motion length: {original_raw_frames} frames.")

    assert original_raw_frames > 0
    original_subsampled_display_count = raw_positions[::subsample_factor].shape[0]
    num_timesteps = min(original_subsampled_display_count, target_subsampled_frames)

    print(
        f"Motion will be displayed for {num_timesteps} subsampled frames "
        f"(original subsampled count: {original_subsampled_display_count})."
    )

    if original_raw_frames >= target_raw_frames:
        processed_positions = raw_positions[:target_raw_frames]
        processed_orientations = raw_orientations[:target_raw_frames]
        processed_left_foot_contacts = raw_left_foot_contacts[:target_raw_frames]
        processed_right_foot_contacts = raw_right_foot_contacts[:target_raw_frames]
    else:
        padding_count = target_raw_frames - original_raw_frames

        processed_positions = onp.concatenate(
            (raw_positions, onp.repeat(raw_positions[-1:], padding_count, axis=0)),
            axis=0,
        )
        processed_orientations = onp.concatenate(
            (
                raw_orientations,
                onp.repeat(raw_orientations[-1:], padding_count, axis=0),
            ),
            axis=0,
        )
        processed_left_foot_contacts = onp.concatenate(
            (
                raw_left_foot_contacts,
                onp.repeat(raw_left_foot_contacts[-1:], padding_count, axis=0),
            ),
            axis=0,
        )
        processed_right_foot_contacts = onp.concatenate(
            (
                raw_right_foot_contacts,
                onp.repeat(raw_right_foot_contacts[-1:], padding_count, axis=0),
            ),
            axis=0,
        )

    left_foot_contacts_avg = onp.mean(
        processed_left_foot_contacts.astype(float), axis=1
    )[:, None]
    right_foot_contacts_avg = onp.mean(
        processed_right_foot_contacts.astype(float), axis=1
    )[:, None]

    window_size = 5

    def apply_crossfade(contact_flags: onp.ndarray) -> onp.ndarray:
        smoothed = onp.zeros_like(contact_flags)
        for index in range(len(contact_flags)):
            start_idx = max(0, index - window_size // 2)
            end_idx = min(len(contact_flags), index + window_size // 2 + 1)
            smoothed[index] = onp.mean(contact_flags[start_idx:end_idx])
        return smoothed

    left_foot_contacts_smoothed = apply_crossfade(left_foot_contacts_avg)
    right_foot_contacts_smoothed = apply_crossfade(right_foot_contacts_avg)

    simplified_keypoints = processed_positions[::subsample_factor]
    simplified_keypoints = _apply_source_scaling(
        simplified_keypoints, source_type, scale_factors
    )

    keypoint_orientations = processed_orientations[::subsample_factor]
    left_foot_contact = left_foot_contacts_smoothed[::subsample_factor]
    right_foot_contact = right_foot_contacts_smoothed[::subsample_factor]

    expected_pos_shape = (target_subsampled_frames, N_RETARGET + N_AUX, 3)
    expected_orient_shape = (target_subsampled_frames, N_RETARGET + N_AUX, 3, 3)
    expected_contact_shape = (target_subsampled_frames, 1)

    assert (
        simplified_keypoints.shape == expected_pos_shape
    ), f"Expected positions shape {expected_pos_shape}, got {simplified_keypoints.shape}"
    assert (
        keypoint_orientations.shape == expected_orient_shape
    ), f"Expected orientations shape {expected_orient_shape}, got {keypoint_orientations.shape}"
    assert (
        left_foot_contact.shape == expected_contact_shape
    ), f"Expected left foot contact shape {expected_contact_shape}, got {left_foot_contact.shape}"
    assert (
        right_foot_contact.shape == expected_contact_shape
    ), f"Expected right foot contact shape {expected_contact_shape}, got {right_foot_contact.shape}"

    return (
        simplified_keypoints,
        keypoint_orientations,
        left_foot_contact,
        right_foot_contact,
        num_timesteps,
    )


def save_contact_labels(
    output_path: str,
    left_foot_contact: onp.ndarray,
    right_foot_contact: onp.ndarray,
    num_timesteps: int,
):
    left_contacts = left_foot_contact[:num_timesteps].squeeze(-1)
    right_contacts = right_foot_contact[:num_timesteps].squeeze(-1)
    foot_contacts = onp.stack([left_contacts, right_contacts], axis=-1)
    onp.savez_compressed(output_path, foot_contacts=foot_contacts)
    print(f"Saved contact labels to {output_path} with shape {foot_contacts.shape}")


@functools.lru_cache(maxsize=1)
def _require_pyroki_runtime() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        jax = importlib.import_module("jax")
        jnp = importlib.import_module("jax.numpy")
        jdc = importlib.import_module("jax_dataclasses")
        jaxlie = importlib.import_module("jaxlie")
        jaxls = importlib.import_module("jaxls")
        pk = importlib.import_module("pyroki")
        yourdfpy = importlib.import_module("yourdfpy")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Retargeting requires the PyRoki environment with JAX, PyRoki, and "
            "yourdfpy installed. Run the retargeting steps with the pyroki_python "
            "interpreter passed to scripts/retarget_amass_to_robot.sh."
        ) from exc

    return jax, jnp, jdc, jaxlie, jaxls, pk, yourdfpy


def make_solve_retargeting(
    robot_config: dict[str, Any],
    human_retarget_names: list[str],
    link_names: list[str],
):
    jax, jnp, jdc, jaxlie, jaxls, pk, _ = _require_pyroki_runtime()
    hand_aux_offset = jnp.array(robot_config["hand_aux_offset"])
    torso_aux_offset = jnp.array(robot_config["torso_aux_offset"])
    torso_link_idx = link_names.index(robot_config["torso_link_name"])
    downweight_hips = bool(robot_config.get("downweight_hips", False))
    joints_to_move_less_names = robot_config.get("joints_to_move_less") or []

    @jaxls.Cost.create_factory
    def joint_vel_limit_cost(var_values, var_joints_curr, var_joints_prev, max_vel, dt, weight):
        joints_curr = var_values[var_joints_curr]
        joints_prev = var_values[var_joints_prev]
        joint_vel = (joints_curr - joints_prev) / dt
        excess_vel = jnp.maximum(jnp.abs(joint_vel) - max_vel, 0.0)
        return excess_vel.flatten() * weight

    @jaxls.Cost.create_factory
    def foot_contact_cost(
        var_values,
        var_Ts_world_root_curr,
        var_Ts_world_root_prev,
        var_robot_cfg_curr,
        var_robot_cfg_prev,
        robot,
        left_foot_contact,
        right_foot_contact,
        joint_retarget_indices,
        foot_indices,
        weight,
    ):
        T_world_root_curr = var_values[var_Ts_world_root_curr]
        T_world_root_prev = var_values[var_Ts_world_root_prev]
        robot_cfg_curr = var_values[var_robot_cfg_curr]
        robot_cfg_prev = var_values[var_robot_cfg_prev]

        T_root_link_curr = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg_curr))
        T_root_link_prev = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg_prev))
        T_world_link_curr = T_world_root_curr @ T_root_link_curr
        T_world_link_prev = T_world_root_prev @ T_root_link_prev

        left_ankle_idx, right_ankle_idx, left_foot_idx, right_foot_idx = foot_indices

        left_ankle_robot_idx = joint_retarget_indices[left_ankle_idx]
        right_ankle_robot_idx = joint_retarget_indices[right_ankle_idx]
        left_foot_robot_idx = joint_retarget_indices[left_foot_idx]
        right_foot_robot_idx = joint_retarget_indices[right_foot_idx]

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

        left_ankle_vel = left_ankle_curr - left_ankle_prev
        right_ankle_vel = right_ankle_curr - right_ankle_prev
        left_foot_vel = left_foot_curr - left_foot_prev
        right_foot_vel = right_foot_curr - right_foot_prev

        left_ankle_toe_z_diff = left_ankle_curr[2] - left_foot_curr[2]
        right_ankle_toe_z_diff = right_ankle_curr[2] - right_foot_curr[2]

        left_contact_weight = left_foot_contact[0]
        right_contact_weight = right_foot_contact[0]

        left_ankle_vel_cost = left_contact_weight * left_ankle_vel
        right_ankle_vel_cost = right_contact_weight * right_ankle_vel
        left_foot_vel_cost = left_contact_weight * left_foot_vel
        right_foot_vel_cost = right_contact_weight * right_foot_vel
        left_z_consistency_cost = left_contact_weight * left_ankle_toe_z_diff
        right_z_consistency_cost = right_contact_weight * right_ankle_toe_z_diff

        return (
            jnp.concatenate(
                [
                    left_ankle_vel_cost.flatten(),
                    right_ankle_vel_cost.flatten(),
                    left_foot_vel_cost.flatten(),
                    right_foot_vel_cost.flatten(),
                    jnp.array([left_z_consistency_cost]),
                    jnp.array([right_z_consistency_cost]),
                ]
            )
            * weight
        )

    @jaxls.Cost.create_factory
    def foot_tilt_cost(
        var_values,
        var_Ts_world_root,
        var_robot_cfg,
        robot,
        left_foot_contact,
        right_foot_contact,
        joint_retarget_indices,
        foot_indices,
        weight,
    ):
        T_world_root = var_values[var_Ts_world_root]
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_link = T_world_root @ T_root_link

        left_ankle_idx, right_ankle_idx, _, _ = foot_indices
        left_ankle_robot_idx = joint_retarget_indices[left_ankle_idx]
        right_ankle_robot_idx = joint_retarget_indices[right_ankle_idx]

        left_foot_ori = T_world_link.rotation().as_matrix()[left_ankle_robot_idx]
        right_foot_ori = T_world_link.rotation().as_matrix()[right_ankle_robot_idx]

        left_contact_weight = left_foot_contact[0]
        right_contact_weight = right_foot_contact[0]

        left_tilt_residual = left_contact_weight * (left_foot_ori[2, 2] - 1.0)
        right_tilt_residual = right_contact_weight * (right_foot_ori[2, 2] - 1.0)

        return (
            jnp.concatenate(
                [
                    jnp.array([left_tilt_residual]),
                    jnp.array([right_tilt_residual]),
                ]
            )
            * weight
        )

    @jdc.jit
    def solve_retargeting(
        robot,
        robot_coll,
        target_keypoints,
        target_orientations,
        left_foot_contact,
        right_foot_contact,
        joint_retarget_indices,
        retarget_mask,
        weights,
        subsample_factor=1,
        input_fps=30.0,
    ):
        del robot_coll

        n_retarget = len(joint_retarget_indices)
        timesteps = target_keypoints.shape[0]

        foot_indices = jnp.array(
            [
                human_retarget_names.index("left_ankle"),
                human_retarget_names.index("right_ankle"),
                human_retarget_names.index("left_foot"),
                human_retarget_names.index("right_foot"),
            ]
        )

        joints_to_move_less_indices = None
        if joints_to_move_less_names:
            joints_to_move_less_indices = jnp.array(
                [
                    robot.joints.actuated_names.index(name)
                    for name in joints_to_move_less_names
                ]
            )

        class SimplifiedJointsScaleVar(
            jaxls.Var,
            default_factory=lambda: jnp.ones((n_retarget, n_retarget)),
        ): ...

        var_joints = robot.joint_var_cls(jnp.arange(timesteps))
        var_Ts_world_root = jaxls.SE3Var(jnp.arange(timesteps))
        var_joints_scale = SimplifiedJointsScaleVar(jnp.zeros(timesteps))

        root_init_se3_list = []
        for timestep in range(timesteps):
            root_pos_t = target_keypoints[timestep, 0, :]
            root_rot_t = target_orientations[timestep, 0, :, :]
            root_init_se3_list.append(
                jaxlie.SE3.from_rotation_and_translation(
                    jaxlie.SO3.from_matrix(root_rot_t), root_pos_t
                )
            )

        root_init_values = jaxlie.SE3(
            jnp.stack([se3.wxyz_xyz for se3 in root_init_se3_list])
        )

        @jaxls.Cost.create_factory
        def retargeting_cost(var_values, var_Ts_world_root, var_robot_cfg, var_joints_scale, keypoints):
            robot_cfg = var_values[var_robot_cfg]
            T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
            T_world_root = var_values[var_Ts_world_root]
            T_world_link = T_world_root @ T_root_link

            target_pos = keypoints[:N_RETARGET, :]
            robot_pos = T_world_link.translation()[jnp.array(joint_retarget_indices)]

            delta_target = target_pos[:, None] - target_pos[None, :]
            delta_robot = robot_pos[:, None] - robot_pos[None, :]

            position_scale = var_values[var_joints_scale][..., None]
            residual_position_delta = (
                (delta_target - delta_robot * position_scale)
                * (1 - jnp.eye(delta_target.shape[0])[..., None])
                * retarget_mask[..., None]
            )

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
                * retarget_mask
            )

            return (
                jnp.concatenate(
                    [
                        residual_position_delta.flatten(),
                        residual_angle_delta.flatten(),
                    ]
                )
                * weights["local_alignment"]
            )

        @jaxls.Cost.create_factory
        def scale_regularization(var_values, var_joints_scale):
            res_0 = (var_values[var_joints_scale] - 1.0).flatten() * 1.0
            res_1 = (
                var_values[var_joints_scale] - var_values[var_joints_scale].T
            ).flatten() * 100.0
            res_2 = jnp.clip(-var_values[var_joints_scale], min=0).flatten() * 100.0
            return jnp.concatenate([res_0, res_1, res_2])

        @jaxls.Cost.create_factory
        def pc_alignment_cost(
            var_values,
            var_Ts_world_root,
            var_robot_cfg,
            var_joints_scale,
            keypoints,
        ):
            del var_joints_scale

            T_world_root = var_values[var_Ts_world_root]
            robot_cfg = var_values[var_robot_cfg]
            T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
            T_world_link = T_world_root @ T_root_link
            link_pos = T_world_link.translation()[joint_retarget_indices]

            left_wrist = human_retarget_names.index("left_wrist")
            left_wrist_idx = joint_retarget_indices[left_wrist]
            link_pos_left_wrist = T_world_link.translation()[left_wrist_idx]
            link_rot_mat_left_wrist = T_world_link.rotation().as_matrix()[left_wrist_idx]
            left_hand_aux_pos = (
                link_pos_left_wrist + link_rot_mat_left_wrist @ hand_aux_offset
            )

            right_wrist = human_retarget_names.index("right_wrist")
            right_wrist_idx = joint_retarget_indices[right_wrist]
            link_pos_right_wrist = T_world_link.translation()[right_wrist_idx]
            link_rot_mat_right_wrist = T_world_link.rotation().as_matrix()[right_wrist_idx]
            right_hand_aux_pos = (
                link_pos_right_wrist + link_rot_mat_right_wrist @ hand_aux_offset
            )

            link_pos_torso = T_world_link.translation()[torso_link_idx]
            link_rot_mat_torso = T_world_link.rotation().as_matrix()[torso_link_idx]
            torso_aux_pos = link_pos_torso + link_rot_mat_torso @ torso_aux_offset

            link_pos_with_aux = jnp.concatenate(
                [
                    link_pos,
                    left_hand_aux_pos[None, :],
                    right_hand_aux_pos[None, :],
                    torso_aux_pos[None, :],
                ],
                axis=0,
            )

            keypoint_pos = keypoints
            keypoint_pos = keypoint_pos.at[-2, :].set(keypoint_pos[-2, :] / 4.0)
            link_pos_with_aux = link_pos_with_aux.at[-2, :].set(
                link_pos_with_aux[-2, :] / 4.0
            )
            keypoint_pos = keypoint_pos.at[-3, :].set(keypoint_pos[-3, :] / 4.0)
            link_pos_with_aux = link_pos_with_aux.at[-3, :].set(
                link_pos_with_aux[-3, :] / 4.0
            )
            keypoint_pos = keypoint_pos.at[-6, :].set(keypoint_pos[-6, :] / 4.0)
            link_pos_with_aux = link_pos_with_aux.at[-6, :].set(
                link_pos_with_aux[-6, :] / 4.0
            )
            keypoint_pos = keypoint_pos.at[-7, :].set(keypoint_pos[-7, :] / 4.0)
            link_pos_with_aux = link_pos_with_aux.at[-7, :].set(
                link_pos_with_aux[-7, :] / 4.0
            )

            if downweight_hips:
                keypoint_pos = keypoint_pos.at[1, :].set(keypoint_pos[1, :] / 4.0)
                link_pos_with_aux = link_pos_with_aux.at[1, :].set(
                    link_pos_with_aux[1, :] / 4.0
                )
                keypoint_pos = keypoint_pos.at[2, :].set(keypoint_pos[2, :] / 4.0)
                link_pos_with_aux = link_pos_with_aux.at[2, :].set(
                    link_pos_with_aux[2, :] / 4.0
                )

            return (link_pos_with_aux - keypoint_pos).flatten() * weights[
                "global_alignment"
            ]

        @jaxls.Cost.create_factory
        def root_smoothness(var_values, var_Ts_world_root, var_Ts_world_root_prev):
            return (
                var_values[var_Ts_world_root].inverse()
                @ var_values[var_Ts_world_root_prev]
            ).log().flatten() * weights["root_smoothness"]

        costs = [
            retargeting_cost(
                var_Ts_world_root,
                var_joints,
                var_joints_scale,
                target_keypoints,
            ),
            scale_regularization(var_joints_scale),
            pk.costs.limit_cost(
                jax.tree.map(lambda value: value[None], robot),
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
            joint_vel_limit_cost(
                robot.joint_var_cls(jnp.arange(1, timesteps)),
                robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
                20.0,
                subsample_factor / input_fps,
                weights["joint_vel_limit"],
            ),
        ]

        if joints_to_move_less_indices is not None:
            costs.append(
                pk.costs.rest_cost(
                    var_joints,
                    var_joints.default_factory()[None],
                    jnp.full(var_joints.default_factory().shape, 0.02)
                    .at[joints_to_move_less_indices]
                    .set(weights["joint_rest_penalty"])[None],
                )
            )

        for timestep in range(1, timesteps):
            costs.append(
                foot_contact_cost(
                    jaxls.SE3Var(timestep),
                    jaxls.SE3Var(timestep - 1),
                    robot.joint_var_cls(timestep),
                    robot.joint_var_cls(timestep - 1),
                    robot,
                    left_foot_contact[timestep],
                    right_foot_contact[timestep],
                    joint_retarget_indices,
                    foot_indices,
                    weights["foot_contact"],
                )
            )

        for timestep in range(timesteps):
            costs.append(
                foot_tilt_cost(
                    jaxls.SE3Var(timestep),
                    robot.joint_var_cls(timestep),
                    robot,
                    left_foot_contact[timestep],
                    right_foot_contact[timestep],
                    joint_retarget_indices,
                    foot_indices,
                    weights["foot_tilt"],
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
                        var_joints,
                        var_Ts_world_root.with_value(root_init_values),
                        var_joints_scale,
                    ]
                ),
                termination=jaxls.TerminationConfig(max_iterations=800),
            )
        )

        return solution[var_Ts_world_root], solution[var_joints]

    return solve_retargeting


def _load_robot(
    robot_config: dict[str, Any],
    urdf_override: str | None,
    mesh_override: str | None,
):
    _, _, _, _, _, pk, yourdfpy = _require_pyroki_runtime()
    urdf_path = urdf_override or robot_config["urdf_path"]
    mesh_dir = mesh_override or robot_config["mesh_dir"]
    urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=mesh_dir)
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = None
    return urdf, robot, robot_coll


def _load_current_motion(
    motion_path: str,
    args: argparse.Namespace,
    robot_config: dict[str, Any],
):
    return load_motion_data(
        motion_path,
        args.source_type,
        args.subsample_factor,
        args.target_raw_frames,
        robot_config["scale_factors"],
    )


def _run_save_contacts_only(
    args: argparse.Namespace,
    motion_paths: list[str],
    robot_config: dict[str, Any],
):
    print(
        "Running in save-contacts-only mode. Extracting foot contact labels from source motions."
    )

    contacts_dir = (
        args.contacts_dir
        if args.contacts_dir
        else os.path.join(args.keypoints_folder_path, "contacts")
    )
    os.makedirs(contacts_dir, exist_ok=True)

    for index, motion_path in enumerate(motion_paths):
        print(
            f"Processing motion {index + 1}/{len(motion_paths)}: {os.path.basename(motion_path)}"
        )

        base_filename = os.path.splitext(os.path.basename(motion_path))[0]
        output_filename = f"{base_filename}_contacts.npz"
        output_path = os.path.join(contacts_dir, output_filename)

        if args.skip_existing and os.path.exists(output_path):
            print(f"Output file {output_filename} already exists, skipping...")
            continue

        _, _, left_foot_contact, right_foot_contact, num_timesteps = _load_current_motion(
            motion_path, args, robot_config
        )
        save_contact_labels(
            output_path, left_foot_contact, right_foot_contact, num_timesteps
        )


def _run_visualization(
    args: argparse.Namespace,
    motion_paths: list[str],
    robot_config: dict[str, Any],
    urdf: Any,
    robot: Any,
    robot_coll: Any,
    human_retarget_names: list[str],
    joint_retarget_indices: Any,
    retarget_mask: Any,
    solve_retargeting: Any,
):
    _, _, _, _, _, pk, _ = _require_pyroki_runtime()
    viser = importlib.import_module("viser")
    ViserUrdf = importlib.import_module("viser.extras").ViserUrdf

    current_motion_index = 0
    (
        simplified_keypoints,
        keypoint_orientations,
        left_foot_contact,
        right_foot_contact,
        num_timesteps,
    ) = _load_current_motion(motion_paths[current_motion_index], args, robot_config)

    server = viser.ViserServer()
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    playing = server.gui.add_checkbox("playing", True)
    timestep_slider = server.gui.add_slider(
        "timestep", 0, num_timesteps - 1 if num_timesteps > 0 else 0, 1, 0
    )

    def reset_timeline_callback(_):
        timestep_slider.value = 0

    reset_timeline_button = server.gui.add_button("Reset Timeline")
    reset_timeline_button.on_click(reset_timeline_callback)

    weights = pk.viewer.WeightTuner(
        server,
        robot_config["weights"],
    )

    Ts_world_root, joints = None, None

    def generate_trajectory():
        nonlocal Ts_world_root, joints
        gen_button.disabled = True
        retarget_next_button.disabled = True
        Ts_world_root, joints = solve_retargeting(
            robot=robot,
            robot_coll=robot_coll,
            target_keypoints=simplified_keypoints,
            target_orientations=keypoint_orientations,
            left_foot_contact=left_foot_contact,
            right_foot_contact=right_foot_contact,
            joint_retarget_indices=joint_retarget_indices,
            retarget_mask=retarget_mask,
            weights=weights.get_weights(),
            subsample_factor=args.subsample_factor,
            input_fps=args.input_fps,
        )
        gen_button.disabled = False
        retarget_next_button.disabled = False

    gen_button = server.gui.add_button("Retarget!")
    gen_button.on_click(lambda _: generate_trajectory())

    def retarget_next_motion(_):
        nonlocal current_motion_index, Ts_world_root, joints, num_timesteps
        nonlocal simplified_keypoints, keypoint_orientations
        nonlocal left_foot_contact, right_foot_contact

        current_motion_index = (current_motion_index + 1) % len(motion_paths)
        (
            simplified_keypoints,
            keypoint_orientations,
            left_foot_contact,
            right_foot_contact,
            num_timesteps,
        ) = _load_current_motion(motion_paths[current_motion_index], args, robot_config)

        timestep_slider.max = num_timesteps - 1 if num_timesteps > 0 else 0
        timestep_slider.value = 0
        Ts_world_root, joints = None, None
        generate_trajectory()

    retarget_next_button = server.gui.add_button("Retarget Next")
    retarget_next_button.on_click(retarget_next_motion)

    generate_trajectory()
    assert Ts_world_root is not None and joints is not None
    del human_retarget_names

    while True:
        with server.atomic():
            if playing.value and num_timesteps > 0:
                timestep_slider.value = (timestep_slider.value + 1) % num_timesteps
            timestep = timestep_slider.value

        try:
            base_frame.wxyz = onp.array(Ts_world_root.wxyz_xyz[timestep][:4])
            base_frame.position = onp.array(Ts_world_root.wxyz_xyz[timestep][4:])
            urdf_vis.update_cfg(onp.array(joints[timestep]))
            server.scene.add_point_cloud(
                "/target_keypoints",
                onp.array(simplified_keypoints[timestep]),
                onp.array((0, 0, 255))[None].repeat(
                    simplified_keypoints.shape[1], axis=0
                ),
                point_size=0.01,
            )
        except Exception:
            pass

        time.sleep(args.subsample_factor / args.input_fps)


def _run_batch(
    args: argparse.Namespace,
    motion_paths: list[str],
    robot_config: dict[str, Any],
    robot: Any,
    robot_coll: Any,
    joint_retarget_indices: Any,
    retarget_mask: Any,
    solve_retargeting: Any,
):
    print("Running in non-visualize mode. Retargeting all motions and saving to disk.")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for index, motion_path in enumerate(motion_paths):
        print(
            f"Processing motion {index + 1}/{len(motion_paths)}: {os.path.basename(motion_path)}"
        )

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
        ) = _load_current_motion(motion_path, args, robot_config)

        Ts_world_root, joints = solve_retargeting(
            robot=robot,
            robot_coll=robot_coll,
            target_keypoints=simplified_keypoints,
            target_orientations=keypoint_orientations,
            left_foot_contact=left_foot_contact,
            right_foot_contact=right_foot_contact,
            joint_retarget_indices=joint_retarget_indices,
            retarget_mask=retarget_mask,
            weights=robot_config["weights"],
            subsample_factor=args.subsample_factor,
            input_fps=args.input_fps,
        )

        results_to_save = {
            "base_frame_pos": onp.array(Ts_world_root.wxyz_xyz[:num_timesteps, 4:]),
            "base_frame_wxyz": onp.array(Ts_world_root.wxyz_xyz[:num_timesteps, :4]),
            "joint_angles": onp.array(joints[:num_timesteps]),
        }

        onp.savez_compressed(output_path, **results_to_save)
        print(f"Saved retargeted motion to {output_path}")


def parse_args(default_robot_config: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simplified Humanoid Retargeting")
    parser.add_argument(
        "--robot-config",
        type=str,
        default=default_robot_config,
        required=default_robot_config is None,
        help="Path to the YAML robot retarget config.",
    )
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
        default=None,
        help="Optional override for the robot URDF path.",
    )
    parser.add_argument(
        "--mesh-dir",
        type=str,
        default=None,
        help="Optional override for the robot mesh directory.",
    )
    parser.add_argument(
        "--subsample-factor",
        type=int,
        default=1,
        help="Subsample factor for the keypoints.",
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
        help="Skip processing motions that already have retargeted output files.",
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
        help="Skip retargeting and only save processed foot contact labels.",
    )
    parser.add_argument(
        "--contacts-dir",
        type=str,
        default=None,
        help="Directory to save contact labels. Defaults to {keypoints_folder_path}/contacts",
    )
    parser.add_argument(
        "--input-fps",
        type=float,
        default=30.0,
        help="FPS of the input keypoint data before subsampling.",
    )
    return parser.parse_args()


def main(default_robot_config: str | None = None):
    args = parse_args(default_robot_config=default_robot_config)
    robot_config = load_robot_config(args.robot_config)

    motion_paths = sorted(glob.glob(os.path.join(args.keypoints_folder_path, "*.npy")))
    if not motion_paths:
        print(f"No .npy files found in {args.keypoints_folder_path}. Exiting.")
        return

    if args.save_contacts_only:
        _run_save_contacts_only(args, motion_paths, robot_config)
        return

    _, jnp, _, _, _, _, _ = _require_pyroki_runtime()
    urdf, robot, robot_coll = _load_robot(robot_config, args.urdf_path, args.mesh_dir)
    link_names = list(robot.links.names)
    human_retarget_names, joint_retarget_indices = get_humanoid_retarget_indices(
        link_names, robot_config
    )
    retarget_mask = build_retarget_mask(human_retarget_names)
    joint_retarget_indices = jnp.array(joint_retarget_indices)
    retarget_mask = jnp.array(retarget_mask)
    solve_retargeting = make_solve_retargeting(
        robot_config, human_retarget_names, link_names
    )

    if args.visualize:
        _run_visualization(
            args,
            motion_paths,
            robot_config,
            urdf,
            robot,
            robot_coll,
            human_retarget_names,
            joint_retarget_indices,
            retarget_mask,
            solve_retargeting,
        )
        return

    _run_batch(
        args,
        motion_paths,
        robot_config,
        robot,
        robot_coll,
        joint_retarget_indices,
        retarget_mask,
        solve_retargeting,
    )


if __name__ == "__main__":
    main()