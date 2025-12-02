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
Compose a mujoco xml file from a set of neutral joints and joint parents.

Unverified example script - adapt for your own skeleton.

This script composes a mujoco xml file from a set of neutral joints and joint parents.
The neutral joints are in global coordinates
The script also plots the skeleton from the neutral joints and joint parents.
The script also generates the mujoco xml file.
"""

import torch
import os
import matplotlib.pyplot as plt
from protomotions.utils.rotations import quat_from_two_vectors

# TODO: add way to specify joint damping, stiffness, etc.
# also width of the body cylinders
# also leave more room between the cylinders from to
# also axis sign for left and right
# also gear of joints

NEUTRAL_JOINTS_FILE = ""
# N by 3 tensor
NEUTRAL_PARENTS_FILE = ""
# e.g. tensor([-1,  0,  1,  2,  3,  4,  5,  4,  7,  8,  9, 10, 10, 12,  4, 14, 15, 16,
#         17, 17, 19,  0, 21, 22, 23,  0, 25, 26, 27])

joint_names = [
    "Hips",
    # list your joint names here
]

# get the parent dir of current file
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
neutral_joints = torch.load(os.path.join(parent_dir, NEUTRAL_JOINTS_FILE))
joint_parents = torch.load(os.path.join(parent_dir, NEUTRAL_PARENTS_FILE))


dead_joint_names = ["RightHandEnd", "LeftHandEnd", "RightHandThumb1", "LeftHandThumb1"]

# create box instead of capsule if in this list
non_leaf_box_width_dict = {
    "RightFoot": (0.1, 0.05),
    "LeftFoot": (0.1, 0.05),
}
leaf_box_size_dict = {
    "RightToeBase": (0.1, 0.1, 0.05),
    "LeftToeBase": (0.1, 0.1, 0.05),
    "Head": (0.1, 0.1, 0.2),
}

print(torch.round(neutral_joints * 100) / 100)

# rotate the skeleton 90 degrees around the x axis from y up to z up
neutral_joints = neutral_joints.to(torch.float32)
neutral_joints = neutral_joints @ torch.tensor([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])

print(joint_parents)

# plot the skeleton from neutral joints and joint parents

# Create 3D plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# Plot joints as points
joints = neutral_joints.numpy()
ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c="b", marker="o")

# Plot bones as lines connecting joints based on parent relationships
for i in range(len(joint_parents)):
    if joint_parents[i] >= 0:  # Skip root joint which has parent -1
        start = joints[joint_parents[i]]
        end = joints[i]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], "r-")

# Set labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Skeleton Visualization")

# same scale
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

plt.show()

# generate the mujoco xml file
# use the neutral joints and joint parents
# note that neutral joints are in global coordinates, but pos in mujoco are in local coordinates

# leaf joints
leaf_joints = [i for i in range(len(joint_parents)) if i not in joint_parents]
print("leaf joints: ", leaf_joints)

# Convert neutral joints to local coordinates by subtracting parent positions
local_positions = torch.zeros_like(neutral_joints)  # (29, 3)
for i in range(len(joint_parents)):
    if joint_parents[i] >= 0:  # Skip root
        local_positions[i] = neutral_joints[i] - neutral_joints[joint_parents[i]]
    else:
        local_positions[i] = neutral_joints[i]

# record the joints in the order they got added to the xml
joints_in_xml_order = []
all_ctrl_dof_names = []

# Generate XML string
xml = '<mujoco model="humanoid">\n'
xml += '  <compiler coordinate="local"/>\n'
xml += "  <default>\n"
xml += '    <joint damping="0.0" armature="0.01" stiffness="0.0" limited="true"/>\n'
xml += '    <geom conaffinity="1" condim="3" contype="7" margin="0.001" rgba="0.8 0.6 .4 1"/>\n'
xml += "  </default>\n"
xml += "  <asset>\n"
xml += '    <texture type="skybox" builtin="gradient" rgb1=".4 .5 .6" rgb2="0 0 0" width="100" height="100"/>\n'
xml += '    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>\n'
xml += '    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>\n'
xml += '    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>\n'
xml += '    <material name="geom" texture="texgeom" texuniform="true"/>\n'
xml += "  </asset>\n"
xml += "  <worldbody>\n"
xml += '    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>\n'


def add_body(joint_idx, depth=0):
    global xml
    indent = "    " * (depth + 1)
    pos = local_positions[joint_idx]
    pos_str = f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"

    # Add opening body tag
    xml += f"{indent}<body name='{joint_names[joint_idx]}' pos='{pos_str}'>\n"
    joints_in_xml_order.append(joint_names[joint_idx])

    children = [i for i in range(len(joint_parents)) if joint_parents[i] == joint_idx]

    if joint_parents[joint_idx] == -1:
        # Add freejoint for root
        xml += f"{indent}  <freejoint name='{joint_names[joint_idx]}'/>\n"
    else:
        # Add 3 hinge joints for the non-root joint
        names = ["x", "y", "z"]
        full_names = [joint_names[joint_idx] + "_" + name for name in names]
        axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for name, axis in zip(full_names, axes):
            xml += f"{indent}  <joint name='{name}' type='hinge' pos='0 0 0' axis='{axis[0]:.4f} {axis[1]:.4f} {axis[2]:.4f}' stiffness='800' damping='80' armature='0.02' range='-180.0000 180.0000'/>\n"
            all_ctrl_dof_names.append(name)

    if joint_parents[joint_idx] == -1:
        # add geom for root
        xml += f"{indent}  <geom type='sphere' size='0.05' pos='0 0 0' density='1000' material='geom'/>\n"
    elif joint_idx not in leaf_joints:
        # add geom for each non-leaf joint
        capsule_size = 0.05
        capsule_from = torch.tensor([0, 0, 0])
        # get the mean of the children positions
        capsule_to = torch.mean(
            torch.stack([local_positions[child] for child in children]), dim=0
        )
        capsule_to = capsule_to * 0.9  # offset the capsule to the center of the body

        if joint_names[joint_idx] not in non_leaf_box_width_dict:
            xml += f"{indent}  <geom type='capsule' size='{capsule_size}' contype='1' conaffinity='1' fromto='{capsule_from[0]:.4f} {capsule_from[1]:.4f} {capsule_from[2]:.4f} {capsule_to[0]:.4f} {capsule_to[1]:.4f} {capsule_to[2]:.4f}' density='1000' material='geom'/>\n"
        else:
            box_width = non_leaf_box_width_dict[joint_names[joint_idx]]
            pos = (capsule_from + capsule_to) / 2
            box_length = torch.norm(capsule_to - capsule_from)

            # get the axis of the capsule
            axis = (capsule_to - capsule_from) / torch.norm(capsule_to - capsule_from)
            quat = quat_from_two_vectors(
                axis, torch.tensor([1.0, 0.0, 0.0]), w_last=False
            )
            quat = quat.squeeze(0)
            quat_str = f"{quat[0]:.4f} {quat[1]:.4f} {quat[2]:.4f} {quat[3]:.4f}"

            size_str = f"{box_length/2:.4f} {box_width[0]/2:.4f} {box_width[1]/2:.4f}"

            pos_str = f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"

            # x_y_z_weight = [
            #     abs(torch.dot(capsule_to - capsule_from, torch.tensor([1., 0., 0.]))),
            #     abs(torch.dot(capsule_to - capsule_from, torch.tensor([0., 1., 0.]))),
            #     abs(torch.dot(capsule_to - capsule_from, torch.tensor([0., 0., 1.])))
            # ]
            # x_or_y_or_z = x_y_z_weight.index(max(x_y_z_weight))
            # if x_or_y_or_z == 0:
            #     size = f"{box_length} {box_width[0]} {box_width[1]}"
            # elif x_or_y_or_z == 1:
            #     size = f"{box_width[0]} {box_length} {box_width[1]}"
            # else:
            #     size = f"{box_width[0]} {box_width[1]} {box_length}"

            xml += f"{indent}  <geom type='box' size='{size_str}' pos='{pos_str}' quat='{quat_str}' density='1000' material='geom'/>\n"
    else:
        if joint_names[joint_idx] not in leaf_box_size_dict:
            # add a sphere geom for the leaf joint
            xml += f"{indent}  <geom type='sphere' size='0.05' pos='0 0 0' density='1000' material='geom'/>\n"
        else:
            box_width = leaf_box_size_dict[joint_names[joint_idx]]
            size_str = f"{box_width[0]/2:.4f} {box_width[1]/2:.4f} {box_width[2]/2:.4f}"
            xml += f"{indent}  <geom type='box' size='{size_str}' pos='0 0 0' quat='1 0 0 0' density='1000' material='geom'/>\n"

    # Recursively add children
    for child in children:
        if joint_names[child] not in dead_joint_names:
            add_body(child, depth + 1)

    # Add closing body tag
    xml += f"{indent}</body>\n"


# Start from root (index 0)
add_body(0)

xml += "  </worldbody>\n"

xml += "  <actuator>\n"
for name in all_ctrl_dof_names:
    xml += f"      <motor name='{name}' joint='{name}' gear='500'/>\n"
xml += "  </actuator>\n"

xml += "  <contact>\n"
xml += "      <exclude body1='Spine' body2='Spine2'/>\n"
xml += "      <exclude body1='Spine1' body2='Spine3'/>\n"
xml += "      <exclude body1='Spine2' body2='Neck'/>\n"
xml += "      <exclude body1='Spine3' body2='Head'/>\n"
xml += "      <exclude body1='Spine' body2='Spine3'/>\n"
xml += "      <exclude body1='Spine1' body2='Neck'/>\n"
xml += "      <exclude body1='Spine2' body2='Head'/>\n"
xml += "      <exclude body1='RightArm' body2='Spine3'/>\n"
xml += "      <exclude body1='LeftArm' body2='Spine3'/>\n"
xml += "      <exclude body1='RightArm' body2='Spine2'/>\n"
xml += "      <exclude body1='LeftArm' body2='Spine2'/>\n"
xml += "  </contact>\n"

xml += "  <sensor/>\n"
xml += "  <size njmax='700' nconmax='700'/>\n"

xml += "</mujoco>"

print("joints in xml order: ", joints_in_xml_order)
# get the index of the joints in the xml order
joints_in_xml_order_idx = [joint_names.index(name) for name in joints_in_xml_order]
print("joints in xml order idx: ", joints_in_xml_order_idx)
# input("Press Enter to continue...")

# write to file
with open("protomotions/data/assets/mjcf/out.xml", "w") as f:
    f.write(xml)
