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
"""
Batch-Optimized Forward Kinematics and Related Utilities for MJCF Models.

This module provides functions for performing batched forward kinematics (FK)
calculations on articulated rigid body systems following MuJoCo MJCF XML definition.
It leverages PyTorch for efficient computation on GPUs or CPUs.

**Key Functionalities:**

- Parsing kinematic structure (parent indices, local transforms, joint axes)
  from an MJCF file (`extract_kinematic_info`).
- Converting batched MuJoCo joint position vectors (`qpos`) into intermediate
  root positions and relative joint rotation matrices (`extract_transforms_from_qpos`).
  Handles bodies with multiple hinge degrees of freedom (DOFs), supporting
  both independent hinge angle and coupled exponential map representations for 3-DOF joints.
- Reconstructing `qpos` vectors from root positions and relative joint rotation
  matrices (`extract_qpos_from_transforms`), with options for decomposing
  3-DOF rotations ('euler_xyz', 'exp_map').
- Computing forward kinematics (world positions and rotations) from the
  intermediate transforms (`compute_forward_kinematics_from_transforms`).
- Calculating relative joint rotation matrices from global body rotations
  (`compute_joint_rot_mats_from_global_mats`).
- Estimating Cartesian linear and angular velocities from time series of
  poses using finite differences
  (`compute_cartesian_velocity`, `compute_angular_velocity`,
  `compute_kinematics_velocities`).
- High-level functions combining these steps to perform FK and velocity
  computation directly from `qpos` (`fk_batch_mjcf_with_velocities`) or
  intermediate transforms (`fk_from_transforms_with_velocities`), returning
  results packaged in a `RobotState` object.
- Test functions for verifying FK results against MuJoCo and checking the
  invertibility of the `qpos` <-> transform conversions.

**Assumptions:**

- The root body must have a free joint (either via `<freejoint/>` tag or
  `<joint type="free"/>`).
- Hinge joints must have their axes defined.
- Bodies are expected to have either 1 (hinge) or 3 DOFs associated with them
  for qpos reconstruction (`extract_qpos_from_transforms`).
- The MuJoCo `qpos` layout is assumed: [root_pos(3), root_quat_wxyz(4), joint_angles(...)].
"""

import torch
import numpy as np
from dm_control import mjcf
import logging
import sys
from typing import Dict, Tuple, Optional, List
from protomotions.utils.config_builder import ConfigBuilder
import re

import mujoco
from protomotions.utils.rotations import (
    quat_from_angle_axis,
    quaternion_to_matrix,
    matrix_to_quaternion,
    quat_conjugate,
    quat_mul_norm,
    quat_angle_axis,
    quat_identity_like,
    get_euler_xyz,
    exp_map_to_quat,
    quat_to_exp_map,
    angle_from_matrix_axis,
)

from dataclasses import dataclass

from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    StateConversion,
)


@dataclass
class KinematicInfo(ConfigBuilder):
    """Stores kinematic information extracted from an MJCF model.

    Contains structural data about the robot including body hierarchy, joint limits,
    local transformations, and degrees of freedom information. All fields are populated
    by parsing the MJCF file.
    """

    body_names: List[str]
    dof_names: List[str]  # exclude floating root DOFs
    parent_indices: List[int]
    local_pos: torch.Tensor
    local_rot_ref_mat: torch.Tensor
    hinge_axes_map: Dict[
        int, torch.Tensor
    ]  # Maps body_idx to its hinge axes [NumHingeDOFs, 3]
    nq: int  # Dimension of qpos (includes root free joint)
    nv: int  # Dimension of qvel (includes root free joint)
    num_bodies: int
    num_dofs: int
    dof_limits_lower: torch.Tensor  # Lower joint limits for all DOFs
    dof_limits_upper: torch.Tensor  # Upper joint limits for all DOFs


@dataclass
class ControlInfo(ConfigBuilder):
    """Stores control information extracted from an MJCF model.

    Contains PD control parameters and joint limits for a single degree of freedom.
    All values are extracted from the MJCF file's actuator and joint definitions.
    """

    stiffness: Optional[float] = None
    damping: Optional[float] = None
    armature: Optional[float] = None
    friction: Optional[float] = None
    effort_limit: Optional[float] = None
    velocity_limit: Optional[float] = None


def build_body_ids_tensor(
    all_body_names: List[str], subset_body_names: List[str], device: torch.device
) -> torch.Tensor:
    """
    Build a tensor of body IDs based on the provided body names.

    Args:
        all_body_names (List[str]): List of all body names.
        subset_body_names (List[str]): List of subset body names.
        device (torch.device): Device to store the tensor on.

    Returns:
        torch.Tensor: Tensor containing indices corresponding to the body names.
    """
    if subset_body_names is None:
        return torch.tensor([], dtype=torch.long, device=device)

    body_ids: List[int] = []
    for body_name in subset_body_names:
        body_id = all_body_names.index(body_name)
        assert body_id != -1, f"Body part {body_name} not found in {all_body_names}"
        body_ids.append(body_id)
    body_ids = torch.tensor(body_ids, dtype=torch.long, device=device)
    return body_ids


def extract_kinematic_info(mjcf_path: str) -> KinematicInfo:
    """
    Extracts kinematic information needed for FK from an mjcf XML file.
    Handles multiple hinge joints per body.

    Args:
        mjcf_path (str): Path to the MJCF XML file.

    Returns:
        KinematicInfo: An object containing the kinematic information.
            See KinematicInfo dataclass for more details.

    Raises:
        ValueError: If any body is unnamed or the root body configuration is invalid.
        AssertionError: If internal consistency checks fail (e.g., joint axis missing).
    """

    mjcf_model = mjcf.from_path(mjcf_path)

    # Check if angles are in degrees or radians (default is degrees in MuJoCo)
    angle_unit = getattr(mjcf_model.compiler, "angle", None)
    # Default to degrees unless explicitly set to 'radian'
    angle_to_radians = 1.0 if angle_unit == "radian" else np.pi / 180.0

    bodies = []
    non_root_dof_names = []
    parent_indices_list = []
    local_pos_list = []
    local_quat_list = []
    dof_limits_lower_list = []
    dof_limits_upper_list = []

    # Store info per hinge DOF
    hinge_axes_map_dict = {}

    body_name_to_idx = {}
    root_processed = False

    worldbody = mjcf_model.worldbody

    def _traverse(mjcf_body, parent_idx):
        nonlocal root_processed
        body_name = mjcf_body.name
        if not body_name:
            raise ValueError(
                f"All bodies need a name for FK. Found unnamed body parented to {parent_idx}"
            )

        body_idx = len(bodies)
        body_name_to_idx[body_name] = body_idx
        bodies.append(body_name)
        parent_indices_list.append(parent_idx)

        pos_val = np.array(
            mjcf_body.pos if mjcf_body.pos is not None else [0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        # MJCF quat is always WXYZ
        quat_mjcf = np.array(
            mjcf_body.quat if mjcf_body.quat is not None else [1.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        local_pos_list.append(pos_val)
        local_quat_list.append(quat_mjcf)

        if body_idx == 0:  # Root Body
            assert np.allclose(
                local_quat_list[0], [1.0, 0.0, 0.0, 0.0]
            ), "Root body must have a quat of [1.0, 0.0, 0.0, 0.0]"
            assert not root_processed, "Multiple root bodies"
            has_free_joint_element = any(j.type == "free" for j in mjcf_body.joint)
            has_freejoint_tag = mjcf_body.freejoint is not None
            assert (
                has_free_joint_element or has_freejoint_tag
            ), "Root body must have a free joint defined."
            root_processed = True
        else:  # Non-Root Bodies
            current_hinge_axes = []  # List of axes [x,y,z] of each hinge DOF at this body

            joints = mjcf_body.joint
            # if len(joints) == 0:
            #     logging.warning(f"Body {body_name} has no joints, skipping.")
            if len(joints) > 0:
                assert (
                    len(joints) == 1 or len(joints) == 3
                ), f"Body {body_name} has {len(joints)} joints, expected 1 or 3"
                for joint in joints:
                    assert (
                        joint.axis is not None
                    ), f"Hinge joint '{joint.name}' for body '{body_name}' has no axis defined."
                    axis_val = np.array(joint.axis, dtype=np.float32)
                    current_hinge_axes.append(axis_val)
                    non_root_dof_names.append(joint.name)

                    # Extract joint limits (convert to radians if needed)
                    if joint.range is not None:
                        # joint.range is [lower, upper]
                        dof_limits_lower_list.append(joint.range[0] * angle_to_radians)
                        dof_limits_upper_list.append(joint.range[1] * angle_to_radians)
                    else:
                        # No limits specified, use large values
                        dof_limits_lower_list.append(-1e10)
                        dof_limits_upper_list.append(1e10)

                current_hinge_axes_np = np.array(current_hinge_axes, dtype=np.float32)
                hinge_axes_map_dict[body_idx] = torch.from_numpy(current_hinge_axes_np)

        child_bodies = mjcf_body.body
        for child in child_bodies:
            _traverse(child, body_idx)

    assert worldbody.body, "MJCF worldbody contains no bodies."
    root_body = worldbody.body[0]
    _traverse(root_body, -1)
    assert root_processed, "Failed to identify/process root body."

    local_quat_numpy = np.array(
        local_quat_list, dtype=np.float32
    )  # (N, 4) concatenated quats
    local_rot_ref_mat_tensor = quaternion_to_matrix(
        torch.from_numpy(local_quat_numpy), w_last=False
    )
    local_pos_numpy = np.array(
        local_pos_list, dtype=np.float32
    )  # (N, 3) concatenated positions

    num_articulated_dofs = sum(
        len(axes_tensor) for axes_tensor in hinge_axes_map_dict.values()
    )

    # Convert joint limits to tensors
    dof_limits_lower = torch.tensor(dof_limits_lower_list, dtype=torch.float32)
    dof_limits_upper = torch.tensor(dof_limits_upper_list, dtype=torch.float32)

    return KinematicInfo(
        body_names=bodies,
        dof_names=non_root_dof_names,
        parent_indices=parent_indices_list,
        local_pos=torch.from_numpy(local_pos_numpy),
        local_rot_ref_mat=local_rot_ref_mat_tensor,
        hinge_axes_map=hinge_axes_map_dict,
        nq=num_articulated_dofs + 7,  # 7 for root free joint (3 pos, 4 quat)
        nv=num_articulated_dofs + 6,  # 6 for root free joint (3 vel, 3 ang_vel)
        num_bodies=len(bodies),
        num_dofs=len(non_root_dof_names),
        dof_limits_lower=dof_limits_lower,
        dof_limits_upper=dof_limits_upper,
    )


def extract_control_info(
    mjcf_path: str, override_control_info: Optional[Dict[str, ControlInfo]] = None
) -> Dict[str, ControlInfo]:
    """
    Extracts control information (stiffness, damping, etc.) from an MJCF XML file.

    This function traverses the MJCF model and extracts control parameters for all DOFs.
    It looks for control parameters in the following order of priority:
    1. Joint-level attributes (stiffness, damping, armature, frictionloss)
    2. Actuator-specific parameters if defined
    3. Default values if parameters are not specified
    4. Override values if specified in the override_control_info dictionary

    Args:
        mjcf_path (str): Path to the MJCF XML file.
        override_control_info (Optional[Dict[str, ControlInfo]]): Override control information for specific joints.
    Returns:
        ControlInfo: An object containing the control information for all DOFs.

    Raises:
        ValueError: If the MJCF file cannot be parsed or if required information is missing.
    """
    try:
        mjcf_model = mjcf.from_path(mjcf_path)
    except Exception as e:
        raise ValueError(f"Failed to parse MJCF file {mjcf_path}: {e}")

    # Default values for control parameters
    DEFAULT_STIFFNESS = None
    DEFAULT_DAMPING = None
    DEFAULT_ARMATURE = None
    DEFAULT_FRICTION = None
    DEFAULT_EFFORT_LIMIT = None

    control_info = {}

    def _extract_joint_control_params(joint_name, joint):
        """Extract control parameters from a joint element."""
        # Extract stiffness and damping from joint attributes
        stiffness = getattr(joint, "stiffness", DEFAULT_STIFFNESS)
        damping = getattr(joint, "damping", DEFAULT_DAMPING)
        armature = getattr(joint, "armature", DEFAULT_ARMATURE)
        friction = getattr(joint, "frictionloss", DEFAULT_FRICTION)
        velocity_limit = None

        # Extract actuator force range if available
        effort_limit = getattr(joint, "actuatorfrcrange", DEFAULT_EFFORT_LIMIT)

        # Convert to float if they're strings (common in MJCF)
        if isinstance(stiffness, str):
            stiffness = float(stiffness)
        if isinstance(damping, str):
            damping = float(damping)
        if isinstance(armature, str):
            armature = float(armature)
        if isinstance(friction, str):
            friction = float(friction)
        if isinstance(effort_limit, str):
            # Parse "min max" format
            try:
                _, max_val = map(float, effort_limit.split())
                effort_limit = max_val
            except Exception:
                effort_limit = DEFAULT_EFFORT_LIMIT
        if isinstance(effort_limit, list):
            effort_limit = effort_limit[1]
        if isinstance(effort_limit, np.ndarray):
            effort_limit = effort_limit[1]

        if override_control_info is not None:
            for (
                joint_expr,
                override_joint_control_info,
            ) in override_control_info.items():
                if re.fullmatch(joint_expr, joint_name):
                    print(f"Overriding control info for {joint_name} with {joint_expr}")
                    if override_joint_control_info.stiffness is not None:
                        stiffness = override_joint_control_info.stiffness
                    if override_joint_control_info.damping is not None:
                        damping = override_joint_control_info.damping
                    if override_joint_control_info.armature is not None:
                        armature = override_joint_control_info.armature
                    if override_joint_control_info.friction is not None:
                        friction = override_joint_control_info.friction
                    if override_joint_control_info.effort_limit is not None:
                        effort_limit = override_joint_control_info.effort_limit
                    if override_joint_control_info.velocity_limit is not None:
                        velocity_limit = override_joint_control_info.velocity_limit

        dof_control_info = ControlInfo(
            stiffness=stiffness,
            damping=damping,
            armature=armature,
            friction=friction,
            effort_limit=effort_limit,
            velocity_limit=velocity_limit,
        )

        return dof_control_info

    def _traverse_bodies_for_control(mjcf_body):
        """Traverse bodies to find joints and extract control parameters."""
        # Process joints in current body
        for joint in mjcf_body.joint:
            if joint.type != "free":  # Skip free joints (root)
                joint_name = joint.name
                if joint_name:
                    # Extract control parameters
                    dof_control_info = _extract_joint_control_params(joint_name, joint)
                    control_info[joint_name] = dof_control_info

        # Recursively process child bodies
        for child_body in mjcf_body.body:
            _traverse_bodies_for_control(child_body)

    # Start traversal from worldbody
    if hasattr(mjcf_model, "worldbody") and mjcf_model.worldbody:
        for body in mjcf_model.worldbody.body:
            _traverse_bodies_for_control(body)

    # Check if we found any DOFs
    if not control_info:
        raise ValueError(f"No DOFs found in MJCF file {mjcf_path}")

    # Convert lists to tensors
    return control_info


# --- Helper: Extract Transforms from qpos ---


@torch.jit.script
def extract_transforms_from_qpos_non_root_ignore_fixed_helper(
    hinge_axes_map: Dict[int, torch.Tensor],
    qpos_non_root: torch.Tensor,
    qpos_is_exp_map_on_3dof_joints: bool = False,
) -> torch.Tensor:
    """
    Helper function to extract transforms from qpos, ignoring bodies with no DOFs.
    """

    device = qpos_non_root.device
    dtype = qpos_non_root.dtype
    B = qpos_non_root.shape[0]
    hinge_axes_map = {k: v.to(device).to(dtype) for k, v in hinge_axes_map.items()}

    joint_start = 0
    num_movable_bodies = len(hinge_axes_map.keys())
    assert num_movable_bodies > 0, "No movable bodies found"

    joint_rot_mats = (
        torch.eye(3, device=device, dtype=dtype)
        .reshape(1, 1, 3, 3)
        .expand(B, num_movable_bodies, 3, 3)
        .clone()
    )

    # Process each body's DOFs
    for i, (body_idx, axes) in enumerate(hinge_axes_map.items()):
        assert body_idx != 0, "Hinge DOF mapped to root body index 0"

        num_body_dofs = len(axes)
        qpos_indices = torch.arange(
            joint_start, joint_start + num_body_dofs, device=device, dtype=torch.long
        )
        joint_start += num_body_dofs

        if num_body_dofs == 3 and qpos_is_exp_map_on_3dof_joints:
            # Get exponential map vector (3 components)
            exp_map_vec = qpos_non_root[:, qpos_indices]
            quat_k = exp_map_to_quat(exp_map_vec, w_last=False)
            joint_rot_mats[:, i, :, :] = quaternion_to_matrix(quat_k, w_last=False)
        else:
            # Get all hinge angles at once
            all_hinge_angles = qpos_non_root[:, qpos_indices]

            all_hinge_quats_wxyz = quat_from_angle_axis(
                all_hinge_angles, axes.unsqueeze(0).expand(B, -1, -1), w_last=False
            )  # (B, NumHingeDOFs, 4)
            # Convert XYZW quaternions to matrices
            all_hinge_rot_mats = quaternion_to_matrix(
                all_hinge_quats_wxyz, w_last=False
            )  # (B, NumHingeDOFs, 3, 3)

            # Accumulate rotations onto the corresponding body's matrix
            # Assumes the order in hinge_* lists matches the desired composition order (usually true based on XML)
            for k in range(num_body_dofs):
                single_dof_rot_mat = all_hinge_rot_mats[:, k, :, :]
                # Note: we are overwriting the identity init here
                joint_rot_mats[:, i, :, :] = torch.matmul(
                    joint_rot_mats[:, i, :, :].clone(), single_dof_rot_mat
                )

    return joint_rot_mats


def extract_transforms_from_qpos_non_root(
    kinematic_info: KinematicInfo,
    qpos_non_root: torch.Tensor,
    qpos_is_exp_map_on_3dof_joints: bool = False,
) -> torch.Tensor:
    """
    Extracts COMPOUNDED relative joint rotation matrices from qpos.
    Handles multiple hinge DOFs per body, including 3-DOF joints represented
    either as independent hinges or coupled exponential maps.

    Args:
        kinematic_info: Dictionary containing pre-parsed kinematic structure
                        information from `extract_kinematic_info`.
        qpos: Batch of joint positions (B, Nq-7). The layout must match the
              order defined by MuJoCo (hinge angles...).
        qpos_is_exp_map_on_3dof_joints: If True, assumes qpos values for 3-DOF joints
                                      represent a single 3D exponential map vector.
                                      If False, assumes they are three independent
                                      hinge angles around the body's joint axes.

    Returns:
        torch.Tensor:
            - joint_rot_mats (B, Nb, 3, 3): Relative rotation matrix for each body.
                Index 0 contains the root's world rotation matrix, which is identity in this case.
    """

    device = qpos_non_root.device
    dtype = qpos_non_root.dtype

    B = qpos_non_root.shape[0]
    Nb = kinematic_info.num_bodies
    hinge_axes_map = kinematic_info.hinge_axes_map
    assert (
        qpos_non_root.shape[1] == kinematic_info.nq - 7
    ), f"qpos_non_root has {qpos_non_root.shape[1]} DOFs, expected {kinematic_info.nq - 7}"

    # Initialize ALL rotation matrices (including root) to identity first
    joint_rot_mats = (
        torch.eye(3, device=device, dtype=dtype)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(B, Nb, 3, 3)
        .clone()
    )

    joint_rot_mats[:, list(hinge_axes_map.keys()), :, :] = (
        extract_transforms_from_qpos_non_root_ignore_fixed_helper(
            hinge_axes_map, qpos_non_root, qpos_is_exp_map_on_3dof_joints
        )
    )

    return joint_rot_mats


def extract_transforms_from_qpos(
    kinematic_info: KinematicInfo,
    qpos: torch.Tensor,
    qpos_is_exp_map_on_3dof_joints: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts root position and COMPOUNDED relative joint rotation matrices from qpos.
    Stores root rotation in the first index of joint_rot_mats.
    Handles multiple hinge DOFs per body, including 3-DOF joints represented
    either as independent hinges or coupled exponential maps.

    Args:
        kinematic_info: Dictionary containing pre-parsed kinematic structure
                        information from `extract_kinematic_info`.
        qpos: Batch of joint positions (B, Nq). The layout must match the
              order defined by MuJoCo (root pos, root quat WXYZ, hinge angles...).
        qpos_is_exp_map_on_3dof_joints: If True, assumes qpos values for 3-DOF joints
                                      represent a single 3D exponential map vector.
                                      If False, assumes they are three independent
                                      hinge angles around the body's joint axes.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - root_pos (B, 3): Root position extracted from qpos.
            - joint_rot_mats (B, Nb, 3, 3): Relative rotation matrix for each body.
                                          Index 0 contains the root's world rotation matrix.
                                          Other indices contain the combined hinge rotations
                                          relative to the parent frame (identity if no hinge).
    """
    device = qpos.device
    dtype = qpos.dtype

    # Extract root pos
    root_pos = qpos[:, 0:3].to(device).to(dtype)

    # Calculate and store root rotation matrix at index 0
    root_quat_wxyz = qpos[:, 3:7].to(device).to(dtype)
    root_quat_wxyz_norm = root_quat_wxyz / torch.linalg.norm(
        root_quat_wxyz, dim=-1, keepdim=True
    )
    root_rot_mat = quaternion_to_matrix(root_quat_wxyz_norm, w_last=False)

    # Calculate and compound rotations for each hinge DOF for non-root bodies
    joint_rot_mats = extract_transforms_from_qpos_non_root(
        kinematic_info, qpos[:, 7:], qpos_is_exp_map_on_3dof_joints
    )

    joint_rot_mats[:, 0, :, :] = root_rot_mat  # Store root rotation here

    return root_pos, joint_rot_mats  # Return only root_pos and the combined mats


# --- Helper: Extract qpos from Transforms (Inverse) ---


def extract_qpos_from_transforms(
    kinematic_info: KinematicInfo,
    root_pos: torch.Tensor,
    joint_rot_mats: torch.Tensor,
    multi_dof_decomposition_method: Optional[str] = None,
) -> torch.Tensor:
    """
    Reconstructs the qpos tensor from root position and relative joint rotations.

    This is the inverse operation of `extract_transforms_from_qpos`.
    It assumes the MuJoCo qpos layout (root pos, root quat WXYZ, hinge angles...).
    Handles bodies with 1 or 3 hinge DOFs. For 3 DOFs, requires specifying
    `multi_dof_decomposition_method`.

    Args:
        kinematic_info: Dictionary containing pre-parsed kinematic structure.
        root_pos (B, 3): Root position.
        joint_rot_mats (B, Nb, 3, 3): Relative rotation matrices for each body.
                                      Index 0 contains the root's world rotation matrix.
                                      Other indices contain the compounded hinge rotations
                                      relative to the parent frame (or identity if no hinges).
        multi_dof_decomposition_method (str, optional): Method to decompose rotation
                                      for bodies with 3 DOFs. Options: 'euler_xyz', 'exp_map'.
                                      Required if any body has 3 DOFs. Defaults to None.

    Returns:
        torch.Tensor (B, Nq): Reconstructed qpos tensor.

    Raises:
        ValueError: If a non-root body has a number of hinge DOFs other than 1 or 3.
        ValueError: If a body has 3 DOFs and `multi_dof_decomposition_method` is
                    not specified or is invalid.
        AssertionError: If internal consistency checks fail (e.g., no hinge DOFs found).
    """
    device = root_pos.device
    dtype = root_pos.dtype
    B = root_pos.shape[0]
    nq = kinematic_info.nq

    qpos = torch.zeros(B, nq, device=device, dtype=dtype)

    # 1. Fill Root Position
    qpos[:, 0:3] = root_pos

    # 2. Fill Root Orientation (WXYZ)
    root_rot_mat = joint_rot_mats[:, 0, :, :]
    root_quat_wxyz = matrix_to_quaternion(root_rot_mat, w_last=False)
    # Normalize quaternion
    root_quat_wxyz_norm = root_quat_wxyz / torch.linalg.norm(
        root_quat_wxyz, dim=-1, keepdim=True
    )
    qpos[:, 3:7] = root_quat_wxyz_norm

    # 3. Fill Hinge Angles
    hinge_axes_map = kinematic_info.hinge_axes_map
    hinge_axes_map = {k: v.to(device).to(dtype) for k, v in hinge_axes_map.items()}
    num_hinge_dofs = nq - 7
    assert num_hinge_dofs > 0, "No hinge DOFs found"

    joint_start = 7

    # Process each body's DOFs
    for body_idx, axes in hinge_axes_map.items():
        num_body_dofs = len(axes)
        qpos_indices = torch.arange(
            joint_start, joint_start + num_body_dofs, device=device, dtype=torch.long
        )
        joint_start += num_body_dofs

        rot_mat_k = joint_rot_mats[:, body_idx, :, :]  # Compounded rotation

        if num_body_dofs == 1:
            axis_k = axes[0]  # Get the single axis
            angle_k = angle_from_matrix_axis(rot_mat_k, axis_k)
            qpos[:, qpos_indices[0]] = angle_k
        elif num_body_dofs == 3:
            # Convert rotation matrix to WXYZ quaternion for decomposition functions
            quat_k = matrix_to_quaternion(rot_mat_k, w_last=False)

            if multi_dof_decomposition_method == "euler_xyz":
                # Get Euler angles (roll, pitch, yaw = X, Y, Z rotations)
                roll, pitch, yaw = get_euler_xyz(quat_k, w_last=False)
                # Assign to qpos indices assuming the order in kinematic_info corresponds to X, Y, Z
                qpos[:, qpos_indices[0]] = roll
                qpos[:, qpos_indices[1]] = pitch
                qpos[:, qpos_indices[2]] = yaw
            elif multi_dof_decomposition_method == "exp_map":
                # Get exponential map vector (3 components)
                exp_map_vec = quat_to_exp_map(quat_k, w_last=False)  # (B, 3)
                # Assign components to qpos indices assuming order correspondence
                qpos[:, qpos_indices[0]] = exp_map_vec[:, 0]
                qpos[:, qpos_indices[1]] = exp_map_vec[:, 1]
                qpos[:, qpos_indices[2]] = exp_map_vec[:, 2]
            else:
                body_name = kinematic_info.body_names[body_idx]
                raise ValueError(
                    f"Invalid 'multi_dof_decomposition_method' ('{multi_dof_decomposition_method}') "
                    f"provided for body '{body_name}' (index {body_idx}). Use 'euler_xyz' or 'exp_map'."
                )
        else:
            # Assert condition for unexpected number of DOFs
            body_name = kinematic_info.body_names[body_idx]
            raise ValueError(
                f"Body '{body_name}' (index {body_idx}) has {num_body_dofs} hinge DOFs. "
                f"Expected 1 or 3 DOFs for decomposition."
            )

    return qpos


# --- Helper: Core FK Computation ---


def compute_forward_kinematics_from_transforms(
    kinematic_info: KinematicInfo,
    root_pos: torch.Tensor,
    joint_rot_mats: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes world poses given kinematic structure and transforms.
    Assumes joint_rot_mats contains root rotation at index 0.

    Args:
        kinematic_info: Dictionary from extract_kinematic_info.
        root_pos (B, 3): Root position.
        joint_rot_mats (B, Nb, 3, 3): Rotation matrices for each body.
                                      Index 0 is root's world rotation.
                                      Other indices are compounded relative joint rotations.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - world_pos (B, Nb, 3): World positions.
            - world_rot_mat (B, Nb, 3, 3): World rotation matrices.
    """

    device = root_pos.device
    dtype = root_pos.dtype

    B = root_pos.shape[0]
    Nb = kinematic_info.num_bodies
    parent_indices = torch.tensor(
        kinematic_info.parent_indices, dtype=torch.long, device=device
    )
    local_pos = (
        kinematic_info.local_pos[None, ...].expand(B, -1, -1).to(device).to(dtype)
    )
    local_rot_ref_mat = (
        kinematic_info.local_rot_ref_mat[None, ...]
        .expand(B, -1, -1, -1)
        .to(device)
        .to(dtype)
    )

    world_pos = torch.zeros(B, Nb, 3, device=device, dtype=dtype)
    world_rot_mat = torch.zeros(B, Nb, 3, 3, device=device, dtype=dtype)

    # FK Loop using matrices
    for i in range(Nb):
        if parent_indices[i] == -1:  # Root body
            world_pos[:, i, :] = root_pos
            # Get root rotation from the combined tensor
            world_rot_mat[:, i, :, :] = joint_rot_mats[:, 0, :, :]
        else:
            parent_idx = parent_indices[i]
            parent_pos_world = world_pos[:, parent_idx, :]
            parent_rot_mat_world = world_rot_mat[:, parent_idx, :, :]

            ref_rot_mat = local_rot_ref_mat[:, i, :, :]
            # Get the compounded hinge joint rotation for this body
            joint_rot_mat = joint_rot_mats[:, i, :, :]

            # Calculate effective local rotation: Ref * Joint
            effective_local_rot = torch.matmul(ref_rot_mat, joint_rot_mat)

            # Calculate world orientation: ParentWorld * EffectiveLocal
            world_rot_mat[:, i, :, :] = torch.matmul(
                parent_rot_mat_world, effective_local_rot
            )

            # Calculate world position: ParentPos + ParentRot * LocalPosOffset
            offset_in_world = torch.matmul(
                parent_rot_mat_world, local_pos[:, i, :, None]
            ).squeeze(-1)
            world_pos[:, i, :] = parent_pos_world + offset_in_world

    return world_pos, world_rot_mat


# --- Helper: Compute Relative Transforms ---


def compute_joint_rot_mats_from_global_mats(
    kinematic_info: KinematicInfo,
    global_rot_mats: torch.Tensor,
) -> torch.Tensor:
    """
    Computes joint rotation matrices relative to the body's reference frame
    from global rotations.

    Calculates J_i = R_ref_i^(-1) * P_p(i)^(-1) * G_i
    where J_i is the joint rotation of body i relative to its reference frame,
    R_ref_i is the reference rotation of body i in the parent frame,
    P_p(i) is the global rotation of its parent p(i), and G_i is the global
    rotation of body i.

    The reference rotation R_ref_i represents the orientation offset specified
    in the MJCF for the body relative to its parent's frame *before* any joint
    rotation is applied. The computed `joint_rot_mats` represents the rotation
    *due to the joint(s)* connecting body `i` to its parent, expressed relative
    to the body's reference frame.

    For the root body (index 0), its reference rotation R_ref_0 is assumed to be
    identity, and its parent is the world frame (also identity rotation).
    Therefore, the "joint" rotation J_0 is simply its global rotation G_0.

    Args:
        kinematic_info: Dictionary from extract_kinematic_info.
        global_rot_mats (torch.Tensor): Global rotation matrices for each body
                                        (B, Nb, 3, 3).

    Returns:
        torch.Tensor (B, Nb, 3, 3): Joint rotation matrices for each body
                                    relative to its reference orientation frame.
                                    Index 0 contains the global rotation of the root.
                                    Other indices contain rotations purely due to joints.
    """
    device = global_rot_mats.device
    dtype = global_rot_mats.dtype
    B, Nb, _, _ = global_rot_mats.shape

    parent_indices = torch.tensor(
        kinematic_info.parent_indices, dtype=torch.long, device=device
    )
    local_rot_ref_mat = (
        kinematic_info.local_rot_ref_mat[None, ...]
        .expand(B, -1, -1, -1)
        .to(device)
        .to(dtype)
    )
    local_rot_ref_mat_inv = local_rot_ref_mat.transpose(-1, -2)  # Precompute inverse

    joint_rot_mats = torch.zeros(B, Nb, 3, 3, device=device, dtype=dtype)

    for i in range(Nb):
        ref_rot_inv = local_rot_ref_mat_inv[:, i, :, :]
        child_global_rot = global_rot_mats[:, i, :, :]

        if parent_indices[i] == -1:  # Root body
            # J_0 = R_ref_0^(-1) * G_0, R_ref_0 is identity
            assert torch.allclose(
                ref_rot_inv, torch.eye(3, device=device, dtype=dtype)
            ), "Root body must have a ref_rot_inv of identity"
            joint_rot_mats[:, i, :, :] = child_global_rot
        else:  # Non-root body
            parent_idx = parent_indices[i]
            parent_global_rot = global_rot_mats[:, parent_idx, :, :]
            parent_global_rot_inv = parent_global_rot.transpose(-1, -2)

            # Calculate effective local rotation: L_i = P_p(i)^(-1) * G_i
            effective_local_rot = torch.matmul(parent_global_rot_inv, child_global_rot)

            # Calculate joint rotation: J_i = R_ref_i^(-1) * L_i
            joint_rot = torch.matmul(ref_rot_inv, effective_local_rot)
            joint_rot_mats[:, i, :, :] = joint_rot

    return joint_rot_mats


# --- Helper: Compute Velocities ---


def compute_cartesian_velocity(
    batched_robot_pos: torch.Tensor,
    fps: int,
) -> torch.Tensor:
    """
    Computes Cartesian velocity by taking the gradient along the time dimension (dim 0).

    Args:
        batched_robot_pos (B, Nb, 3): batched robot positions, where B is the time dimension.
        fps (int): Frames per second.

    Returns:
        torch.Tensor (B, Nb, 3): Cartesian velocities.
    """
    if batched_robot_pos.shape[0] < 2:  # Need at least 2 frames to compute velocity
        return torch.zeros_like(batched_robot_pos)

    # Compute gradient along the time axis (dim=0)
    # torch.gradient returns a list of tensors, one for each dimension gradient is computed over
    pos_grad = torch.gradient(batched_robot_pos, dim=0)[0]
    velocity = pos_grad / (1.0 / fps)

    return velocity


def compute_angular_velocity(
    batched_robot_rot_mats: torch.Tensor,
    fps: int,
) -> torch.Tensor:
    """
    Computes angular velocity from batched rotation matrices using quaternion differentiation.

    Args:
        batched_robot_rot_mats (B, Nb, 3, 3): batched rotation matrices, B is time dimension.
        fps (int): Frames per second.

    Returns:
        torch.Tensor (B, Nb, 3): Angular velocities (axis * angle / dt).
    """
    if batched_robot_rot_mats.shape[0] < 2:
        # Return zero velocity if less than 2 frames
        vel_shape = batched_robot_rot_mats.shape[:-2] + (3,)
        return torch.zeros(
            vel_shape,
            device=batched_robot_rot_mats.device,
            dtype=batched_robot_rot_mats.dtype,
        )

    # Convert matrices to quaternions (use consistent w_last convention)
    batched_robot_quats = matrix_to_quaternion(batched_robot_rot_mats, w_last=True)

    # Calculate difference quaternion between consecutive frames
    # q_diff = q_{t+1} * q_t^{-1}
    quat_t = batched_robot_quats[:-1]
    quat_t_plus_1 = batched_robot_quats[1:]
    quat_t_inv = quat_conjugate(quat_t, w_last=True)
    diff_quat = quat_mul_norm(quat_t_plus_1, quat_t_inv, w_last=True)

    # Pad the difference quaternion array to match the original time dimension length
    # Use identity rotation for the first frame's difference
    identity_quat = quat_identity_like(diff_quat[0:1], w_last=True)
    diff_quat_padded = torch.cat([identity_quat, diff_quat], dim=0)

    # Extract angle and axis from the difference quaternions
    diff_angle, diff_axis = quat_angle_axis(diff_quat_padded, w_last=True)

    # Calculate angular velocity: axis * angle / dt
    angular_velocity = diff_axis * diff_angle.unsqueeze(-1) / (1.0 / fps)

    return angular_velocity


# --- Main FK Functions ---


def compute_kinematics_velocities(
    batched_robot_pos: torch.Tensor,
    batched_robot_rot_mats: torch.Tensor,
    fps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute linear and angular velocities from poses over time.

    """
    lin_vel = compute_cartesian_velocity(batched_robot_pos, fps)

    ang_vel = compute_angular_velocity(batched_robot_rot_mats, fps)

    return lin_vel, ang_vel


def fk_batch_mjcf_with_velocities(
    kinematic_info: KinematicInfo,
    qpos: torch.Tensor,
    fps: Optional[int] = None,
    compute_velocities: bool = True,
) -> RobotState:
    """
    Performs batched forward kinematics with velocities using PyTorch.

    Args:
        kinematic_info: Dictionary from extract_kinematic_info.
        qpos: Batch of joint positions (B, Nq), expects MuJoCo order.
              (root pos, root quat WXYZ, hinge angles...)
        fps: Frames per second.
        compute_velocities: Whether to compute velocities.

    Returns:

        RobotState:
            rigid_body_pos (B, Nb, 3): World positions of each body.
            rigid_body_rot (B, Nb, 4): World quaternions of each body.
            fps (int): Frames per second.
            rigid_body_vel (B, Nb, 3): World velocities of each body.
            rigid_body_ang_vel (B, Nb, 3): World angular velocities of each body.

    """
    # Extract transforms
    root_pos, joint_rot_mats = extract_transforms_from_qpos(kinematic_info, qpos)

    return fk_from_transforms_with_velocities(
        kinematic_info, root_pos, joint_rot_mats, fps, compute_velocities
    )


def fk_from_transforms_with_velocities(
    kinematic_info: KinematicInfo,
    root_pos: torch.Tensor,
    joint_rot_mats: torch.Tensor,
    fps: Optional[int] = None,
    compute_velocities: bool = True,
) -> RobotState:
    """
    Performs forward kinematics with velocities from root position and joint rotations.

    Args:
        kinematic_info: Dictionary from extract_kinematic_info.
        root_pos (B, 3): Root positions.
        joint_rot_mats (B, Nb, 3, 3): Rotation matrices for each body.
        fps: Frames per second.
        compute_velocities: Whether to compute velocities.

    Returns:

        RobotState:
            rigid_body_pos (B, Nb, 3): World positions of each body.
            rigid_body_rot (B, Nb, 4): World quaternions of each body.
            fps (int): Frames per second.
            rigid_body_vel (B, Nb, 3): World velocities of each body.
            rigid_body_ang_vel (B, Nb, 3): World angular velocities of each body.

    """
    # Compute FK
    world_pos, world_rot_mat = compute_forward_kinematics_from_transforms(
        kinematic_info, root_pos, joint_rot_mats
    )

    # Convert rotation matrices to quaternions
    # RobotState uses XYZW convention
    world_quat = matrix_to_quaternion(world_rot_mat, w_last=True)

    result = RobotState(
        rigid_body_pos=world_pos,
        rigid_body_rot=world_quat,
        state_conversion=StateConversion.COMMON,
    )

    result.fps = fps

    # Compute velocities if requested
    # vels do not care about w_last convention
    if compute_velocities and root_pos.shape[0] > 1:
        assert fps is not None, "fps is required when compute_velocities is True"
        lin_vel, ang_vel = compute_kinematics_velocities(world_pos, world_rot_mat, fps)
        result.rigid_body_vel = lin_vel
        result.rigid_body_ang_vel = ang_vel

    return result


# --- Test Function ---


def test_fk_batch(mjcf_path: str, batch_size: int = 5, num_tests: int = 10):
    """
    Tests the fk_batch_mjcf_with_velocities implementation against MuJoCo.

    Generates random qpos batches, computes FK using both this module's implementation
    and MuJoCo's physics engine, and compares the resulting world positions and rotations.

    Args:
        mjcf_path (str): Path to the MJCF file to test.
        batch_size (int, optional): Number of poses per batch. Defaults to 5.
        num_tests (int, optional): Number of random test batches to run. Defaults to 10.

    Returns:
        bool: True if all test runs pass within tolerance, False otherwise.
    """
    logging.info(f"Loading MJCF: {mjcf_path}")
    mjcf_model = mjcf.from_path(mjcf_path)
    physics = mjcf.Physics.from_mjcf_model(mjcf_model)

    logging.info("Extracting kinematic info...")
    kinematic_info = extract_kinematic_info(mjcf_path)
    nq = physics.model.nq
    nv = physics.model.nv
    nb = kinematic_info.num_bodies
    assert (
        nq == kinematic_info.nq
    ), f"Mismatch in nq: MJCF ({kinematic_info.nq}) vs MuJoCo ({nq})"
    logging.info(f"Model Info: Nq={nq}, Nv={nv}, Nb={nb}")
    logging.info(f"Body Names: {kinematic_info.body_names}")
    logging.info(f"Non-root DOF Names: {kinematic_info.dof_names}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    logging.info(f"Using device: {device}")

    jnt_range = physics.model.jnt_range.copy()
    jnt_limited = physics.model.jnt_limited.astype(bool)
    passed_count = 0

    for test_idx in range(num_tests):
        # logging.info(f"--- Test Run {test_idx + 1}/{num_tests} ---") # Verbose

        qpos_batch_np = np.random.randn(batch_size, nq)

        # Assign hinge values using MuJoCo's jnt_qposadr
        for j_idx in range(physics.model.njnt):
            jnt_type = physics.model.jnt_type[j_idx]
            qpos_adr = physics.model.jnt_qposadr[j_idx]
            if jnt_type == mujoco.mjtJoint.mjJNT_HINGE:
                if qpos_adr >= nq:
                    continue
                if jnt_limited[j_idx]:
                    low, high = jnt_range[j_idx]
                    qpos_batch_np[:, qpos_adr] = np.random.uniform(
                        low, high, size=batch_size
                    )
                else:
                    qpos_batch_np[:, qpos_adr] = np.random.randn(batch_size) * np.pi
            elif jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                # Normalize random quaternion for the free joint
                quat_indices = slice(qpos_adr + 3, qpos_adr + 7)
                qpos_batch_np[:, quat_indices] = qpos_batch_np[
                    :, quat_indices
                ] / np.linalg.norm(
                    qpos_batch_np[:, quat_indices], axis=-1, keepdims=True
                )

        qpos_batch_torch = torch.tensor(qpos_batch_np, device=device, dtype=dtype)

        # --- Reference FK (MuJoCo) ---
        ref_pos = np.zeros((batch_size, nb, 3))
        ref_quat_wxyz = np.zeros((batch_size, nb, 4))  # MuJoCo output is WXYZ
        body_indices_in_mujoco = [
            physics.model.body(name).id for name in kinematic_info.body_names
        ]

        for i in range(batch_size):
            with physics.reset_context():
                physics.data.qpos[:] = qpos_batch_np[i]
                physics.forward()
                ref_pos[i] = physics.data.xpos[body_indices_in_mujoco]
                ref_quat_wxyz[i] = physics.data.xquat[body_indices_in_mujoco]

        ref_pos_torch = torch.tensor(ref_pos, device=device, dtype=dtype)
        ref_quat_wxyz_torch = torch.tensor(ref_quat_wxyz, device=device, dtype=dtype)

        # --- Custom FK (PyTorch) ---

        res = fk_batch_mjcf_with_velocities(
            kinematic_info,
            qpos_batch_torch,
            fps=20,  # dummy fps not used
            compute_velocities=True,
        )
        cust_pos = res.rigid_body_pos
        cust_quat = res.rigid_body_rot

        # --- Comparison ---
        pos_close = torch.allclose(ref_pos_torch, cust_pos, atol=1e-4, rtol=1e-4)

        # Prepare reference quaternion in the same format as output for comparison
        ref_quat_compare = ref_quat_wxyz_torch[
            ..., [1, 2, 3, 0]
        ]  # WXYZ -> XYZW: robot state uses XYZW

        ref_quat_norm = ref_quat_compare / torch.linalg.norm(
            ref_quat_compare, dim=-1, keepdim=True
        )
        cust_quat_norm = cust_quat / torch.linalg.norm(cust_quat, dim=-1, keepdim=True)
        # Compare dot product (handles q == -q)
        quat_dist = torch.abs(torch.sum(ref_quat_norm * cust_quat_norm, dim=-1))
        quat_close = torch.all(quat_dist > 0.9999)

        if not (pos_close and quat_close):
            logging.error(f"❌ FK Test Failed (Run {test_idx + 1})")
            pos_diff = torch.abs(ref_pos_torch - cust_pos).max()
            quat_angle_diff = 2 * torch.acos(torch.clamp(quat_dist.min(), -1.0, 1.0))
            logging.error(
                f"Max pos diff: {pos_diff.item():.6f}, Max quat angle diff: {quat_angle_diff.item():.6f} rad"
            )
        else:
            passed_count += 1

    logging.info("--- FK Test Summary ---")
    logging.info(f"Passed {passed_count}/{num_tests} FK test runs.")
    return passed_count == num_tests


def test_qpos_transform_inverse(
    mjcf_path: str, batch_size: int = 5, num_tests: int = 10
) -> bool:
    """
    Tests if extract_qpos_from_transforms is the inverse of extract_transforms_from_qpos.

    Generates random qpos, converts to transforms, then converts back to qpos,
    and checks if the original and reconstructed qpos values are equivalent.
    Handles potential angle wrapping and quaternion sign differences.

    This test is currently skipped if the model contains non-root bodies with more
    than one hinge DOF, as the `extract_qpos_from_transforms` function requires
    a decomposition method ('euler_xyz' or 'exp_map') for multi-DOF joints, which
    might not perfectly invert the composition in `extract_transforms_from_qpos`
    depending on the method chosen and the original qpos values.

    Args:
        mjcf_path (str): Path to the MJCF file to test.
        batch_size (int, optional): Number of poses per batch. Defaults to 5.
        num_tests (int, optional): Number of random test batches to run. Defaults to 10.

    Returns:
        bool: True if all test runs pass within tolerance or if the test is skipped,
              False otherwise.
    """
    logging.info("--- Testing Qpos <-> Transform Inverse ---")
    kinematic_info = extract_kinematic_info(mjcf_path)
    nq = kinematic_info.nq
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Check for multi-DOF bodies
    hinge_axes_map = kinematic_info.hinge_axes_map
    num_counts = [len(axes) for axes in hinge_axes_map.values()]
    if torch.any(torch.tensor(num_counts) > 1):
        multi_dof_bodies = (
            torch.where(torch.tensor(num_counts) > 1)[0] + 1
        )  # +1 to get original indices
        multi_dof_body_names = [
            kinematic_info.body_names[idx.item()] for idx in multi_dof_bodies
        ]
        logging.warning(
            f"⚠️ Skipping Qpos<->Transform inverse test. Model contains bodies with multiple hinge DOFs: {multi_dof_body_names}."
        )
        return True  # Consider skipped test as passed for overall status

    logging.info(
        "Model has only single-DOF hinge bodies (or no hinges). Proceeding with inverse test."
    )
    passed_count = 0

    for test_idx in range(num_tests):
        # Generate random qpos (normalized quaternion for root)
        qpos_orig_np = np.random.randn(batch_size, nq)
        # Normalize root quat (WXYZ at indices 3-7)
        qpos_orig_np[:, 3:7] = qpos_orig_np[:, 3:7] / np.linalg.norm(
            qpos_orig_np[:, 3:7], axis=-1, keepdims=True
        )
        # Clamp hinge angles to avoid large angle wrapping issues (atan2 range)
        qpos_orig_np[:, 7:] = np.clip(qpos_orig_np[:, 7:], -np.pi * 0.95, np.pi * 0.95)

        qpos_orig_torch = torch.tensor(qpos_orig_np, device=device, dtype=dtype)

        # Forward: qpos -> transforms
        root_pos, joint_rot_mats = extract_transforms_from_qpos(
            kinematic_info, qpos_orig_torch
        )

        # Inverse: transforms -> qpos
        # multi_dof_decomposition_method not needed as we excluded those cases
        qpos_reconstructed = extract_qpos_from_transforms(
            kinematic_info, root_pos, joint_rot_mats
        )

        # Compare
        # Root pos comparison
        pos_close = torch.allclose(
            qpos_orig_torch[:, 0:3], qpos_reconstructed[:, 0:3], atol=1e-5, rtol=1e-5
        )

        # Root quat comparison (WXYZ at 3:7) - use dot product check
        q_orig_norm = qpos_orig_torch[:, 3:7] / torch.linalg.norm(
            qpos_orig_torch[:, 3:7], dim=-1, keepdim=True
        )
        q_rec_norm = qpos_reconstructed[:, 3:7] / torch.linalg.norm(
            qpos_reconstructed[:, 3:7], dim=-1, keepdim=True
        )
        quat_dot = torch.abs(torch.sum(q_orig_norm * q_rec_norm, dim=-1))
        quat_close = torch.all(
            quat_dot > 0.99999
        )  # Check if they represent the same rotation

        # Hinge angle comparison (indices >= 7)
        hinge_angles_orig = qpos_orig_torch[:, 7:]
        hinge_angles_rec = qpos_reconstructed[:, 7:]
        # Handle potential angle wrapping differences using torch.remainder
        angle_diff = hinge_angles_rec - hinge_angles_orig
        angle_diff_wrapped = torch.remainder(angle_diff + np.pi, 2 * np.pi) - np.pi
        hinge_close = torch.allclose(
            angle_diff_wrapped,
            torch.zeros_like(angle_diff_wrapped),
            atol=1e-5,
            rtol=1e-5,
        )

        if pos_close and quat_close and hinge_close:
            passed_count += 1
        else:
            logging.error(
                f"❌ Qpos<->Transform Inverse Test Failed (Run {test_idx + 1})"
            )
            if not pos_close:
                logging.error(
                    f"  Max root pos diff: {torch.abs(qpos_orig_torch[:, 0:3] - qpos_reconstructed[:, 0:3]).max().item():.6f}"
                )
            if not quat_close:
                min_dot = quat_dot.min().item()
                logging.error(
                    f"  Min root quat dot product: {min_dot:.6f} (Angle diff: {2 * np.arccos(min_dot):.6f} rad)"
                )
            if not hinge_close:
                logging.error(
                    f"  Max hinge angle wrapped diff: {torch.abs(angle_diff_wrapped).max().item():.6f}"
                )

    logging.info("--- Qpos<->Transform Inverse Test Summary ---")
    logging.info(f"Passed {passed_count}/{num_tests} inverse test runs.")
    return passed_count == num_tests


# --- __main__ ---
if __name__ == "__main__":
    # Re-setup logger in case it's run as main
    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    has_stream_handler = any(
        isinstance(h, logging.StreamHandler) and h.stream == sys.stdout
        for h in logger.handlers
    )
    if not has_stream_handler:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(log_formatter)
        logger.addHandler(stream_handler)

    try:
        import os

        workspace_root = "."  # Adjust if needed
        # mjcf_path = os.path.join(workspace_root, "protomotions/data/assets/mjcf/rigv1_humanoid.xml")
        # mjcf_path = os.path.join(workspace_root, "protomotions/data/assets/mjcf/g1.xml") # Example single-DOF per body
        mjcf_path = os.path.join(
            workspace_root, "protomotions/data/assets/mjcf/h1_2.xml"
        )

        if os.path.exists(mjcf_path):
            logger.info(f"--- Testing Model ({mjcf_path}) ---")
            fk_passed = test_fk_batch(
                mjcf_path, batch_size=10, num_tests=20
            )  # Increased batch/tests
            inverse_passed = test_qpos_transform_inverse(
                mjcf_path, batch_size=10, num_tests=20
            )
            if fk_passed and inverse_passed:
                logger.info("✅ All tests passed.")
            else:
                logger.error("❌ Some tests failed.")
        else:
            logger.error(f"MJCF file not found: {mjcf_path}")
            logger.warning("Skipping test. Please provide a valid MJCF path.")

    except ImportError as e:
        logger.error(
            f"Import error: {e}. Make sure dm_control, mujoco, and scipy are installed."
        )
    except ValueError as e:
        logger.error(f"Configuration error in MJCF or script: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
