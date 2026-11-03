# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analytic PyRoki FK derivatives, independent of the retargeting objective.

Uses the ancestor/twist construction from agile-soma-retarget, without its
robot-specific keypoints or block-tridiagonal objective assumption.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np


def make_forward_kinematics(robot, link_ids):
    """Return FK(cfg) with an analytic JVP; vmap this for multiple frames.

    Results use PyRoki's [w, x, y, z, px, py, pz] link pose convention. Fixed,
    revolute, prismatic and mimic joints use the robot's own twist/driver tables.
    """
    joints = robot.joints
    parents = np.asarray(joints.parent_indices)
    ancestry = np.zeros((robot.links.num_links, joints.num_joints))
    for link, parent in enumerate(np.asarray(robot.links.parent_joint_indices)):
        while parent != -1:
            ancestry[link, parent] = 1
            parent = parents[parent]
    drivers = np.where(
        np.asarray(joints.mimic_act_indices) == -1,
        np.asarray(joints.actuated_indices),
        np.asarray(joints.mimic_act_indices),
    )
    derivative = np.zeros((joints.num_joints, joints.num_actuated_joints))
    for joint, driver in enumerate(drivers):
        if driver != -1:
            derivative[joint, driver] = np.asarray(joints.mimic_multiplier)[joint]
    ids = jnp.asarray(link_ids)
    mask = jnp.asarray(ancestry[np.asarray(link_ids)])
    driver_map = jnp.asarray(derivative)

    def poses(cfg):
        joint_poses = robot._forward_kinematics_joints(cfg, unroll_fk=True)
        return joint_poses, robot._link_poses_from_joint_poses(joint_poses)[ids]

    @jax.custom_jvp
    def forward(cfg):
        return poses(cfg)[1]

    @forward.defjvp
    def forward_jvp(primals, tangents):
        (cfg,), (velocity,) = primals, tangents
        joint_poses, links = poses(cfg)
        joint_frames = jaxlie.SE3(joint_poses)
        rotation = joint_frames.rotation()
        angular_axes = rotation.apply(joints.twists[:, 3:])
        linear_axes = rotation.apply(joints.twists[:, :3])
        full_velocity = driver_map @ velocity
        omega = (mask * full_velocity[None]) @ angular_axes
        offset = links[:, None, 4:] - joint_frames.translation()[None]
        translation_derivative = jnp.sum(
            (jnp.cross(angular_axes[None], offset) + linear_axes[None])
            * (mask * full_velocity[None])[..., None],
            axis=1,
        )
        # World-frame angular velocity acts on the LEFT of the orientation.
        w, xyz = links[:, :1], links[:, 1:4]
        quaternion_derivative = 0.5 * jnp.concatenate(
            (
                -jnp.sum(omega * xyz, axis=1, keepdims=True),
                w * omega + jnp.cross(omega, xyz),
            ),
            axis=1,
        )
        return links, jnp.concatenate(
            (quaternion_derivative, translation_derivative), axis=1
        )

    return forward
