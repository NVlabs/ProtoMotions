# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BVH parsing utilities and skeleton/animation conversion helpers."""

import re
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation


class BvhNode:
    """Lightweight tree node used to represent parsed BVH hierarchy lines."""

    def __init__(self, value=[], parent=None):
        """Create a node from tokenized BVH line values."""
        self.value = value
        self.children = []
        self.parent = parent
        if self.parent:
            self.parent.add_child(self)

    def add_child(self, item):
        """Attach a child node and set its parent reference."""
        item.parent = self
        self.children.append(item)

    def filter(self, key):
        """Yield direct children whose first token matches `key`."""
        for child in self.children:
            if child.value[0] == key:
                yield child

    def __iter__(self):
        for child in self.children:
            yield child

    def __getitem__(self, key):
        """Return all tokens following `key` from the first matching child node."""
        for child in self.children:
            for index, item in enumerate(child.value):
                if item == key:
                    if index + 1 >= len(child.value):
                        return None
                    else:
                        return child.value[index + 1 :]
        raise IndexError("key {} not found".format(key))

    def __repr__(self):
        return str(" ".join(self.value))

    @property
    def name(self):
        """Joint name for `ROOT`/`JOINT` entries."""
        return self.value[1]


class Bvh:
    """Parsed BVH file with hierarchy graph and per-frame channel values."""

    def __init__(self, data: str, backend: Optional[str] = "graph"):
        """
        Args:
            data: Raw BVH file content.
            backend: Parsing mode. `"graph"` keeps list-based frame storage,
                while `"np"` precomputes a NumPy array and index caches.
        """
        self.data = data
        self.root = BvhNode()
        self.frames = []
        self.backend = backend
        self.tokenize()
        if self.backend == "np":
            # cache important info for quick access later
            self.build_data_array()
        elif self.backend == "graph":
            pass
        else:
            raise ValueError(f"Unknown backend for BVH loading: {backend}")

    def build_data_array(self):
        """Build cached channel indices and contiguous frame data for `"np"` backend."""
        joints = self.get_joints()
        self.joint2idx = dict()
        self.joint2channels = dict()
        cur_idx = 0
        for joint in joints:
            self.joint2idx[joint.value[1]] = cur_idx
            cur_idx += int(joint["CHANNELS"][0])
            self.joint2channels[joint.value[1]] = joint["CHANNELS"][1:]
        self.np_data_array = np.array(self.frames, dtype=np.float32)

    def tokenize(self):
        """Tokenize BVH text and populate hierarchy plus frame values."""
        first_round = []
        accumulator = ""
        for char in self.data:
            if char not in ("\n", "\r"):
                accumulator += char
            elif accumulator:
                first_round.append(re.split("\\s+", accumulator.strip()))
                accumulator = ""
        node_stack = [self.root]
        frame_time_found = False
        node = None
        for item in first_round:
            if frame_time_found:
                self.frames.append(item)
                continue
            key = item[0]
            if key == "{":
                node_stack.append(node)
            elif key == "}":
                node_stack.pop()
            else:
                node = BvhNode(item)
                # print("new node: ", node, "\nparent: ", node_stack[-1])
                node_stack[-1].add_child(node)
            if item[0] == "Frame" and item[1] == "Time:":
                frame_time_found = True

    def search(self, *items):
        """Depth-first search for nodes matching a prefix of tokens."""
        found_nodes = []

        def check_children(node):
            if len(node.value) >= len(items):
                failed = False
                for index, item in enumerate(items):
                    if node.value[index] != item:
                        failed = True
                        break
                if not failed:
                    found_nodes.append(node)
            for child in node:
                check_children(child)

        check_children(self.root)
        return found_nodes

    def get_joints(self):
        """Return all `ROOT`/`JOINT` hierarchy joints in BVH traversal order."""
        joints = []

        def iterate_joints(joint):
            joints.append(joint)
            for child in joint.filter("JOINT"):
                iterate_joints(child)

        iterate_joints(next(self.root.filter("ROOT")))
        return joints

    def get_joints_names(self):
        """Return joint names in the same order as :meth:`get_joints`."""
        joints = []

        def iterate_joints(joint):
            joints.append(joint.value[1])
            for child in joint.filter("JOINT"):
                iterate_joints(child)

        iterate_joints(next(self.root.filter("ROOT")))
        return joints

    def joint_direct_children(self, name):
        """Return direct child joints of the given joint name."""
        joint = self.get_joint(name)
        return [child for child in joint.filter("JOINT")]

    def get_joint_index(self, name):
        """Return hierarchy index of the named joint."""
        return self.get_joints().index(self.get_joint(name))

    def get_joint(self, name):
        """Return hierarchy node for a joint name."""
        found = self.search("ROOT", name)
        if not found:
            found = self.search("JOINT", name)
        if found:
            return found[0]
        raise LookupError("joint not found")

    def joint_offset(self, name, idx=[0, 1, 2]):
        """Return selected `OFFSET` components for a joint."""
        joint = self.get_joint(name)
        offset = joint["OFFSET"]
        if len(offset) < max(idx):
            return None
        return (float(offset[idx[0]]), float(offset[idx[1]]), float(offset[idx[2]]))

    def joint_offset_rot(self, name):
        """Return optional rotational offset components from custom BVH files."""
        return self.joint_offset(name, idx=[3, 4, 5])

    def joint_channels(self, name):
        """Return channel names declared for a joint."""
        if self.backend == "np":
            return self.joint2channels[name]
        else:
            joint = self.get_joint(name)
            return joint["CHANNELS"][1:]

    def get_joint_channels_index(self, joint_name):
        """Return the flattened starting channel index for one joint."""
        if self.backend == "np":
            return self.joint2idx[joint_name]
        else:
            index = 0
            for joint in self.get_joints():
                if joint.value[1] == joint_name:
                    return index
                index += int(joint["CHANNELS"][0])
            raise LookupError("joint not found")

    def get_joint_channel_index(self, joint, channel):
        """Return per-joint channel offset for a specific channel name."""
        channels = self.joint_channels(joint)
        if channel in channels:
            channel_index = channels.index(channel)
        else:
            raise ValueError(f"Channel {channel} not found in {channels}")
        return channel_index

    def frame_joint_channel(self, frame_index, joint, channel, value=None):
        """Return one channel value for one joint at one frame index."""
        joint_index = self.get_joint_channels_index(joint)
        channel_index = self.get_joint_channel_index(joint, channel)
        if channel_index == -1 and value is not None:
            return value
        if self.backend == "np":
            return self.np_data_array[frame_index, joint_index + channel_index]
        else:
            return float(self.frames[frame_index][joint_index + channel_index])

    def frame_joint_channels(self, frame_index, joint, channels, value=None):
        """Get single frame data for on specific joint from multiple specific channels (e.g.
        Xrotation, Yrotation, Zrotation)."""
        values = []
        joint_index = self.get_joint_channels_index(joint)
        if self.backend == "np":
            channel_idx = [
                self.get_joint_channel_index(joint, channel) for channel in channels
            ]
            channel_idx = np.array(channel_idx) + joint_index
            values = self.np_data_array[frame_index, channel_idx]
        else:
            for channel in channels:
                channel_index = self.get_joint_channel_index(joint, channel)
                if channel_index == -1 and value is not None:
                    values.append(value)
                else:
                    values.append(
                        float(self.frames[frame_index][joint_index + channel_index])
                    )
        return values

    def frames_joint_channels(self, joint, channels, value=None):
        """Get all frame data for one joint from multiple channels (e.g. Xrotation, Yrotation,
        Zrotation)."""
        joint_index = self.get_joint_channels_index(joint)
        if self.backend == "np":
            channel_idx = [
                self.get_joint_channel_index(joint, channel) for channel in channels
            ]
            channel_idx = np.array(channel_idx) + joint_index
            all_frames = self.np_data_array[:, channel_idx]
        else:
            all_frames = []
            for frame in self.frames:
                values = []
                for channel in channels:
                    channel_index = self.get_joint_channel_index(joint, channel)
                    if channel_index == -1 and value is not None:
                        values.append(value)
                    else:
                        values.append(float(frame[joint_index + channel_index]))
                all_frames.append(values)
        return all_frames

    def frames_joints_channels(self, joint_names, channels):
        """Get all frames for all specified joints with one specified set of channels."""
        if self.backend != "np":
            raise NotImplementedError("Only np backend is supported for this function")
        joint_indices = [
            (joint_name, self.joint2idx[joint_name]) for joint_name in joint_names
        ]
        data_indices = []
        for joint_name, joint_idx in joint_indices:
            channel_indices = [
                self.get_joint_channel_index(joint_name, channel)
                for channel in channels
            ]
            data_indices.extend(
                [joint_idx + channel_idx for channel_idx in channel_indices]
            )
        all_frames = self.np_data_array[:, data_indices]
        all_frames = all_frames.reshape(-1, len(joint_names), len(channels))
        return all_frames

    def joint_parent(self, name):
        """Return parent joint node, or `None` for the root."""
        joint = self.get_joint(name)
        if joint.parent == self.root:
            return None
        return joint.parent

    def joint_parent_index(self, name):
        """Return parent joint index, or `-1` for the root."""
        joint = self.get_joint(name)
        if joint.parent == self.root:
            return -1
        return self.get_joints().index(joint.parent)

    @property
    def nframes(self):
        """Number of motion frames declared in the BVH header."""
        try:
            return int(next(self.root.filter("Frames:")).value[1])
        except StopIteration:
            raise LookupError("number of frames not found")

    @property
    def frame_time(self):
        """Frame duration in seconds declared in the BVH header."""
        try:
            return float(next(self.root.filter("Frame")).value[2])
        except StopIteration:
            raise LookupError("frame time not found")


class Bone:
    """Container for one skeleton bone and its kinematic metadata."""

    def __init__(self):
        # original bone info
        self.id = None
        self.name = None
        self.orient = np.identity(3)
        self.dof_index = []
        self.channels = []  # bvh only
        self.lb = []
        self.ub = []
        self.parent = None
        self.child = []

        # asf specific
        self.dir = np.zeros(3)
        self.len = 0
        # bvh specific
        self.offset = np.zeros(3)  # default offset for position
        self.offset_rot = None  # rotation for custom nv bvh

        # inferred info
        self.pos = np.zeros(3)
        self.end = np.zeros(3)

    def __repr__(self):
        return f"{self.name}"


class SkeletonBvh:
    """Skeleton structure reconstructed from BVH hierarchy metadata."""

    def __init__(self):
        self.bones = []
        self.name2bone = {}
        self.mass_scale = 1.0
        self.len_scale = 1.0
        self.dof_name = ["x", "y", "z"]
        self.root = None

    def get_bones_names(self):
        """Return bone names in skeleton order."""
        return [x.name for x in self.bones]

    def get_parent_indices(self):
        """Return parent index array aligned with `self.bones`."""
        parent_indices = [-1] * len(self.bones)
        for bone in self.bones:
            if bone.parent:
                parent_indices[bone.id] = bone.parent.id
        return parent_indices

    def get_neutral_joints(self):
        """Return neutral/rest joint positions as a NumPy array `(J, 3)`."""
        joints = []
        for bone in self.bones:
            joints.append(bone.pos)
        joints = np.stack(joints, axis=0)
        return joints

    def load_from_bvh(self, fname, exclude_bones=None, spec_channels=None):
        """Load skeleton hierarchy and rest offsets from a BVH file.

        Args:
            fname: Path to a BVH file.
            exclude_bones: Bone-name substrings to ignore while constructing the
                skeleton.
            spec_channels: Optional per-joint channel overrides.
        """
        if exclude_bones is None:
            exclude_bones = {}
        if spec_channels is None:
            spec_channels = dict()
        with open(fname) as f:
            mocap = Bvh(f.read())

        joint_names = list(
            filter(
                lambda x: all([t not in x for t in exclude_bones]),
                mocap.get_joints_names(),
            )
        )
        dof_ind = {"x": 0, "y": 1, "z": 2}
        self.len_scale = 1.0
        self.root = Bone()
        self.root.id = 0
        self.root.name = joint_names[0]
        self.root.channels = mocap.joint_channels(self.root.name)
        self.root.offset = np.array(mocap.joint_offset(self.root.name)) * self.len_scale
        self.root.offset_rot = mocap.joint_offset_rot(self.root.name)
        if self.root.offset_rot is not None:
            self.root.offset_rot = np.array(self.root.offset_rot)
        # self.root.offset = np.zeros_like(self.root.offset) # TODO: remove this
        self.name2bone[self.root.name] = self.root
        self.bones.append(self.root)
        for i, joint in enumerate(joint_names[1:]):
            bone = Bone()
            bone.id = i + 1
            bone.name = joint
            bone.channels = (
                spec_channels[joint]
                if joint in spec_channels.keys()
                else mocap.joint_channels(joint)
            )
            bone.dof_index = [dof_ind[x[0].lower()] for x in bone.channels]
            bone.offset = np.array(mocap.joint_offset(joint)) * self.len_scale
            bone.offset_rot = mocap.joint_offset_rot(joint)
            if bone.offset_rot is not None:
                bone.offset_rot = np.array(bone.offset_rot)
            bone.lb = [-180.0] * 3
            bone.ub = [180.0] * 3
            self.bones.append(bone)
            self.name2bone[joint] = bone

        # for bone in self.bones:
        # print(bone.name, bone.channels, bone.offset)

        for bone in self.bones[1:]:
            parent_name = mocap.joint_parent(bone.name).name
            if parent_name in self.name2bone.keys():
                bone_p = self.name2bone[parent_name]
                bone_p.child.append(bone)
                bone.parent = bone_p

        self.forward_bvh(self.root)
        for bone in self.bones:
            if len(bone.child) == 0:
                child_vals = [str(node) for node in mocap.get_joint(bone.name).children]
                if "End Site" in child_vals:
                    end_site_idx = child_vals.index("End Site")
                    end_site_offset = mocap.get_joint(bone.name).children[end_site_idx][
                        "OFFSET"
                    ]
                    bone.end = (
                        bone.pos
                        + np.array([float(x) for x in end_site_offset]) * self.len_scale
                    )
                else:
                    pass
            else:
                bone.end = sum([bone_c.pos for bone_c in bone.child]) / len(bone.child)

    def forward_bvh(self, bone):
        """Recursively accumulate absolute joint positions from local offsets."""
        if bone.parent:
            bone.pos = bone.parent.pos + bone.offset
        else:
            bone.pos = bone.offset
        for bone_c in bone.child:
            self.forward_bvh(bone_c)


def load_bvh_animation(
    fname: str,
    skeleton: SkeletonBvh,
    rot_order: Optional[str] = "native",
    backend: Optional[str] = "np",
    return_quat: Optional[bool] = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load motion channels from BVH into root translations and joint rotations.

    Args:
        fname: Full path to the BVH file.
        skeleton: Parsed neutral skeleton built from compatible BVH hierarchy.
        rot_order: Euler order to use for conversion (`"native"` keeps BVH order).
        backend: BVH parser backend (`"np"` or `"graph"`).
        return_quat: If `True`, return quaternions instead of rotation matrices.

    Returns:
        Root translations `(T, 3)` and joint rotations `(T, J, 3, 3)` or
        `(T, J, 4)` when `return_quat=True`.
    """
    with open(fname) as f:
        mocap = Bvh(f.read(), backend=backend)

    # assume all joints are same ordering, load in with native ordering
    root_channels = mocap.joint_channels(skeleton.root.name)
    pos_channels = [
        channel for channel in root_channels if channel.endswith("position")
    ]
    rot_channels = [
        channel for channel in root_channels if channel.endswith("rotation")
    ]

    root_trans = np.array(mocap.frames_joint_channels(skeleton.root.name, pos_channels))

    if backend == "np":
        # NOTE: assumes rot channel ordering is the same for all joints
        joint_eulers = mocap.frames_joints_channels(
            skeleton.get_bones_names(), rot_channels
        )
        joint_eulers = np.deg2rad(joint_eulers)
    elif backend == "graph":
        joint_eulers = []
        for bone in skeleton.bones:
            bone_channels = mocap.joint_channels(bone.name)
            bone_rot_channels = [
                channel for channel in bone_channels if channel.endswith("rotation")
            ]
            assert (
                bone_rot_channels == rot_channels
            ), "Rotation channel ordering is not consistent across joints!"
            # use native rotation order
            euler = np.deg2rad(
                np.array(mocap.frames_joint_channels(bone.name, rot_channels))
            )
            joint_eulers.append(euler)
        joint_eulers = np.stack(joint_eulers, axis=1)
    else:
        raise ValueError(f"Unknown backend for BVH loading: {backend}")

    if rot_order == "native":
        rot_order = ""
        for axis in rot_channels:
            rot_order += axis[0]
    else:
        # need to reorder dims
        ordered_joint_eulers = []
        for axis in rot_order:
            i = rot_channels.index(axis + "rotation")
            ordered_joint_eulers.append(joint_eulers[..., i])
        joint_eulers = np.stack(ordered_joint_eulers, axis=-1)

    rotations = Rotation.from_euler(rot_order, joint_eulers.reshape(-1, 3))
    if return_quat:
        joint_rots = rotations.as_quat(scalar_first=True).reshape(
            joint_eulers.shape[:-1] + (4,)
        )
    else:
        joint_rots = rotations.as_matrix().reshape(joint_eulers.shape[:-1] + (3, 3))

    return root_trans, joint_rots


def bvh_local_to_global_rotations(
    local_rot_mats: torch.Tensor,
    parent_indices: list,
) -> torch.Tensor:
    """Compute global rotation matrices from local rotations via FK (rotations only).

    Args:
        local_rot_mats: (T, J, 3, 3) local rotation matrices.
        parent_indices: Length-J list of parent indices (-1 for root).

    Returns:
        (T, J, 3, 3) global rotation matrices.
    """
    T, J = local_rot_mats.shape[:2]
    global_rot = torch.empty_like(local_rot_mats)
    for i in range(J):
        if parent_indices[i] == -1:
            global_rot[:, i] = local_rot_mats[:, i]
        else:
            global_rot[:, i] = global_rot[:, parent_indices[i]] @ local_rot_mats[:, i]
    return global_rot


def global_rots_to_local_rots(
    global_rot_mats: torch.Tensor,
    parent_indices: list,
) -> torch.Tensor:
    """Convert global rotation matrices back to local (parent-relative).

    Args:
        global_rot_mats: (T, J, 3, 3) global rotation matrices.
        parent_indices: Length-J list of parent indices (-1 for root).

    Returns:
        (T, J, 3, 3) local rotation matrices.
    """
    T, J = global_rot_mats.shape[:2]
    local_rot = torch.empty_like(global_rot_mats)
    for i in range(J):
        if parent_indices[i] == -1:
            local_rot[:, i] = global_rot_mats[:, i]
        else:
            p = parent_indices[i]
            local_rot[:, i] = global_rot_mats[:, p].transpose(-1, -2) @ global_rot_mats[:, i]
    return local_rot


def change_tpose(
    local_rot_mats: torch.Tensor,
    global_rot_offsets: torch.Tensor,
    parent_indices: list,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Re-express BVH local rotations in a standard T-pose convention.

    The BVH zero-pose is bone-axis-aligned (not a natural T-pose). This function
    converts the local rotations so they are relative to the standard T-pose
    defined by global_rot_offsets.

    Args:
        local_rot_mats: (T, J, 3, 3) BVH local rotation matrices.
        global_rot_offsets: (J, 3, 3) per-body global rotation offsets defining
            the standard T-pose in the BVH's coordinate system.
        parent_indices: Length-J list of parent indices (-1 for root).

    Returns:
        Tuple of (new_local_rot_mats, new_global_rot_mats) in the standard
        T-pose convention, both (T, J, 3, 3).
    """
    device, dtype = local_rot_mats.device, local_rot_mats.dtype
    global_rot_offsets = global_rot_offsets.to(device=device, dtype=dtype)

    # BVH FK: local → global
    global_rot_mats = bvh_local_to_global_rotations(local_rot_mats, parent_indices)

    # Apply T-pose offset: new_global = old_global @ offsets^T
    new_global_rot_mats = torch.einsum(
        "t j m n, j o n -> t j m o", global_rot_mats, global_rot_offsets
    )

    # Convert back to local
    new_local_rot_mats = global_rots_to_local_rots(new_global_rot_mats, parent_indices)

    return new_local_rot_mats, new_global_rot_mats

