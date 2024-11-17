# Adapted from https://github.com/zhengyiluo/phc/tree/h1_phc

import os
import sys
import os.path as osp

sys.path.append(os.getcwd())
import torch
from collections import defaultdict


import numpy as np
import smpl_sim.utils.rotation_conversions as tRot
from scipy.spatial.transform import Rotation as sRot
import xml.etree.ElementTree as ETree
from easydict import EasyDict
import scipy.ndimage.filters as filters
import smpl_sim.poselib.core.rotation3d as pRot
from lxml.etree import XMLParser, parse, ElementTree, Element, SubElement
from lxml import etree
from io import BytesIO
import copy
from collections import OrderedDict
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from stl import mesh
import logging
import open3d as o3d

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Humanoid_Batch:

    def __init__(self, cfg, device=torch.device("cpu")):
        self.cfg = cfg
        self.mjcf_file = cfg.asset_file

        parser = XMLParser(remove_blank_text=True)
        tree = parse(
            BytesIO(open(self.mjcf_file, "rb").read()),
            parser=parser,
        )
        self.dof_axis = []
        joints = sorted(
            [
                j.attrib["name"]
                for j in tree.getroot().find("worldbody").findall(".//joint")
            ]
        )
        motors = sorted(
            [m.attrib["name"] for m in tree.getroot().find("actuator").getchildren()]
        )
        assert (len(motors) > 0, "No motors found in the mjcf file")

        self.num_dof = len(motors)
        self.num_extend_dof = self.num_dof

        self.mjcf_data = mjcf_data = self.from_mjcf(self.mjcf_file)
        self.body_names = copy.deepcopy(mjcf_data["node_names"])
        self._parents = mjcf_data["parent_indices"]
        self.body_names_augment = copy.deepcopy(mjcf_data["node_names"])
        self._proper_kinematic_structure = copy.deepcopy(mjcf_data["node_names"])
        self._offsets = mjcf_data["local_translation"][None,].to(device)
        self._local_rotation = mjcf_data["local_rotation"][None,].to(device)
        self.actuated_joints_idx = np.array(
            [self.body_names.index(k) for k, v in mjcf_data["body_to_joint"].items()]
        )

        for m in motors:
            if not m in joints:
                print(m)

        if (
            "type" in tree.getroot().find("worldbody").findall(".//joint")[0].attrib
            and tree.getroot().find("worldbody").findall(".//joint")[0].attrib["type"]
            == "free"
        ):
            for j in tree.getroot().find("worldbody").findall(".//joint")[1:]:
                self.dof_axis.append([int(i) for i in j.attrib["axis"].split(" ")])
            self.has_freejoint = True
        elif (
            not "type" in tree.getroot().find("worldbody").findall(".//joint")[0].attrib
        ):
            for j in tree.getroot().find("worldbody").findall(".//joint"):
                self.dof_axis.append([int(i) for i in j.attrib["axis"].split(" ")])
            self.has_freejoint = True
        else:
            for j in tree.getroot().find("worldbody").findall(".//joint")[6:]:
                self.dof_axis.append([int(i) for i in j.attrib["axis"].split(" ")])
            self.has_freejoint = False

        self.dof_axis = torch.tensor(self.dof_axis)

        for extend_config in cfg.extend_config:
            self.body_names_augment += [extend_config.joint_name]
            self._parents = torch.cat(
                [
                    self._parents,
                    torch.tensor([self.body_names.index(extend_config.parent_name)]).to(
                        device
                    ),
                ],
                dim=0,
            )
            self._offsets = torch.cat(
                [self._offsets, torch.tensor([[extend_config.pos]]).to(device)], dim=1
            )
            self._local_rotation = torch.cat(
                [self._local_rotation, torch.tensor([[extend_config.rot]]).to(device)],
                dim=1,
            )
            self.num_extend_dof += 1

            parent_id = self._proper_kinematic_structure.index(
                extend_config.parent_name
            )
            self._proper_kinematic_structure.insert(
                parent_id + 1, extend_config.joint_name
            )

        self.num_bodies = len(self.body_names)
        self.num_bodies_augment = len(self.body_names_augment)

        self.joints_range = mjcf_data["joints_range"].to(device)
        self._local_rotation_mat = tRot.quaternion_to_matrix(
            self._local_rotation
        ).float()  # w, x, y ,z
        self.load_mesh()

        self.extend_to_proper_mapping = []
        for i, name in enumerate(self._proper_kinematic_structure):
            self.extend_to_proper_mapping.append(self.body_names_augment.index(name))

    def from_mjcf(self, path):
        # function from Poselib:
        tree = ETree.parse(path)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")
        if xml_world_body is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
        # assume this is the root
        xml_body_root = xml_world_body.find("body")
        if xml_body_root is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")

        xml_joint_root = xml_body_root.find("joint")

        node_names = []
        parent_indices = []
        local_translation = []
        local_rotation = []
        joints_range = []
        body_to_joint = OrderedDict()

        # recursively adding all nodes into the skel_tree
        def _add_xml_node(xml_node, parent_index, node_index):
            node_name = xml_node.attrib.get("name")
            # parse the local translation into float list
            pos = np.fromstring(
                xml_node.attrib.get("pos", "0 0 0"), dtype=float, sep=" "
            )
            quat = np.fromstring(
                xml_node.attrib.get("quat", "1 0 0 0"), dtype=float, sep=" "
            )
            node_names.append(node_name)
            parent_indices.append(parent_index)
            local_translation.append(pos)
            local_rotation.append(quat)
            curr_index = node_index
            node_index += 1
            all_joints = xml_node.findall(
                "joint"
            )  # joints need to remove the first 6 joints
            if len(all_joints) == 6:
                all_joints = all_joints[6:]

            for joint in all_joints:
                if not joint.attrib.get("range") is None:
                    joints_range.append(
                        np.fromstring(joint.attrib.get("range"), dtype=float, sep=" ")
                    )
                else:
                    if not joint.attrib.get("type") == "free":
                        joints_range.append([-np.pi, np.pi])
            for joint_node in xml_node.findall("joint"):
                body_to_joint[node_name] = joint_node.attrib.get("name")

            for next_node in xml_node.findall("body"):
                node_index = _add_xml_node(next_node, curr_index, node_index)

            return node_index

        _add_xml_node(xml_body_root, -1, 0)
        assert len(joints_range) == self.num_dof
        return {
            "node_names": node_names,
            "parent_indices": torch.from_numpy(
                np.array(parent_indices, dtype=np.int32)
            ),
            "local_translation": torch.from_numpy(
                np.array(local_translation, dtype=np.float32)
            ),
            "local_rotation": torch.from_numpy(
                np.array(local_rotation, dtype=np.float32)
            ),
            "joints_range": torch.from_numpy(np.array(joints_range)),
            "body_to_joint": body_to_joint,
        }

    def fk_batch(self, pose, trans, convert_to_mat=True, return_full=False, dt=1 / 30):
        device, dtype = pose.device, pose.dtype
        pose_input = pose.clone()
        B, seq_len = pose.shape[:2]
        pose = pose[
            ..., : len(self._parents), :
        ]  # H1 fitted joints might have extra joints

        if convert_to_mat:
            pose_quat = tRot.axis_angle_to_quaternion(pose.clone())
            pose_mat = tRot.quaternion_to_matrix(pose_quat)
        else:
            pose_mat = pose

        if pose_mat.shape != 5:
            pose_mat = pose_mat.reshape(B, seq_len, -1, 3, 3)
        J = pose_mat.shape[2] - 1  # Exclude root
        wbody_pos, wbody_mat = self.forward_kinematics_batch(
            pose_mat[:, :, 1:], pose_mat[:, :, 0:1], trans
        )

        return_dict = EasyDict()

        wbody_rot = tRot.wxyz_to_xyzw(tRot.matrix_to_quaternion(wbody_mat))

        return_dict.global_translation = wbody_pos.clone()
        return_dict.global_rotation_mat = wbody_mat.clone()
        return_dict.global_rotation = wbody_rot

        if return_full:
            return_dict.global_velocity = self._compute_velocity(wbody_pos, dt)
            return_dict.global_angular_velocity = self._compute_angular_velocity(
                wbody_rot, dt
            )
            return_dict.local_rotation = tRot.wxyz_to_xyzw(pose_quat)
            return_dict.global_root_velocity = return_dict.global_velocity[..., 0, :]
            return_dict.global_root_angular_velocity = (
                return_dict.global_angular_velocity[..., 0, :]
            )

            if len(self.cfg.extend_config) > 0:
                return_dict.dof_pos = pose.sum(dim=-1)[
                    ..., 1 : self.num_bodies
                ]  # you can sum it up since unitree's each joint has 1 dof. Last two are for hands. doesn't really matter.
            else:
                if not len(self.actuated_joints_idx) == len(self.body_names):
                    return_dict.dof_pos = pose.sum(dim=-1)[
                        ..., self.actuated_joints_idx
                    ]
                else:
                    return_dict.dof_pos = pose.sum(dim=-1)[..., 1:]

            dof_vel = (return_dict.dof_pos[:, 1:] - return_dict.dof_pos[:, :-1]) / dt
            return_dict.dof_vels = torch.cat([dof_vel, dof_vel[:, -2:-1]], dim=1)
            return_dict.fps = int(1 / dt)

        return return_dict

    def convert_to_proper_kinematic(self, return_dict):
        return_dict.global_translation = return_dict.global_translation[
            ..., self.extend_to_proper_mapping, :
        ]
        return_dict.global_rotation_mat = return_dict.global_rotation_mat[
            ..., self.extend_to_proper_mapping, :, :
        ]
        return_dict.global_rotation = return_dict.global_rotation[
            ..., self.extend_to_proper_mapping, :
        ]
        return_dict.global_velocity = return_dict.global_velocity[
            ..., self.extend_to_proper_mapping, :
        ]
        return_dict.global_angular_velocity = return_dict.global_angular_velocity[
            ..., self.extend_to_proper_mapping, :
        ]
        return return_dict

    def forward_kinematics_batch(self, rotations, root_rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where B = batch size, J = number of joints):
         -- rotations: (B, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (B, 3) tensor describing the root joint positions.
        Output: joint positions (B, J, 3)
        """

        device, dtype = root_rotations.device, root_rotations.dtype
        B, seq_len = rotations.size()[0:2]
        J = self._offsets.shape[1]
        positions_world = []
        rotations_world = []

        expanded_offsets = (
            self._offsets[:, None].expand(B, seq_len, J, 3).to(device).type(dtype)
        )
        # print(expanded_offsets.shape, J)

        for i in range(J):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(root_rotations)
            else:
                jpos = (
                    torch.matmul(
                        rotations_world[self._parents[i]][:, :, 0],
                        expanded_offsets[:, :, i, :, None],
                    ).squeeze(-1)
                    + positions_world[self._parents[i]]
                )
                rot_mat = torch.matmul(
                    rotations_world[self._parents[i]],
                    torch.matmul(
                        self._local_rotation_mat[:, (i) : (i + 1)],
                        rotations[:, :, (i - 1) : i, :],
                    ),
                )
                # rot_mat = torch.matmul(rotations_world[self._parents[i]], rotations[:, :, (i - 1):i, :])
                # print(rotations[:, :, (i - 1):i, :].shape, self._local_rotation_mat.shape)

                positions_world.append(jpos)
                rotations_world.append(rot_mat)

        positions_world = torch.stack(positions_world, dim=2)
        rotations_world = torch.cat(rotations_world, dim=2)
        return positions_world, rotations_world

    @staticmethod
    def _compute_velocity(p, time_delta, guassian_filter=True):
        velocity = np.gradient(p.numpy(), axis=-3) / time_delta
        if guassian_filter:
            velocity = torch.from_numpy(
                filters.gaussian_filter1d(velocity, 2, axis=-3, mode="nearest")
            ).to(p)
        else:
            velocity = torch.from_numpy(velocity).to(p)

        return velocity

    @staticmethod
    def _compute_angular_velocity(r, time_delta: float, guassian_filter=True):
        # assume the second last dimension is the time axis
        diff_quat_data = pRot.quat_identity_like(r).to(r)
        diff_quat_data[..., :-1, :, :] = pRot.quat_mul_norm(
            r[..., 1:, :, :], pRot.quat_inverse(r[..., :-1, :, :])
        )
        diff_angle, diff_axis = pRot.quat_angle_axis(diff_quat_data)
        angular_velocity = diff_axis * diff_angle.unsqueeze(-1) / time_delta
        if guassian_filter:
            angular_velocity = torch.from_numpy(
                filters.gaussian_filter1d(
                    angular_velocity.numpy(), 2, axis=-3, mode="nearest"
                ),
            )
        return angular_velocity

    def load_mesh(self):
        xml_base = os.path.dirname(self.mjcf_file)
        # Read the compiler tag from the g1.xml file to find if there is a meshdir defined
        tree = ETree.parse(self.mjcf_file)
        xml_doc_root = tree.getroot()
        compiler_tag = xml_doc_root.find("compiler")

        if compiler_tag is not None and "meshdir" in compiler_tag.attrib:
            mesh_base = os.path.join(xml_base, compiler_tag.attrib["meshdir"])
        else:
            mesh_base = xml_base

        self.tree = tree = ETree.parse(self.mjcf_file)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")

        xml_assets = xml_doc_root.find("asset")
        all_mesh = xml_assets.findall(".//mesh")

        geoms = xml_world_body.findall(".//geom")

        all_joints = xml_world_body.findall(".//joint")
        all_motors = tree.findall(".//motor")
        all_bodies = xml_world_body.findall(".//body")

        def find_parent(root, child):
            for parent in root.iter():
                for elem in parent:
                    if elem == child:
                        return parent
            return None

        mesh_dict = {}
        mesh_parent_dict = {}

        for mesh_file_node in tqdm(all_mesh):
            mesh_name = mesh_file_node.attrib["name"]
            mesh_file = mesh_file_node.attrib["file"]
            mesh_full_file = osp.join(mesh_base, mesh_file)
            mesh_obj = o3d.io.read_triangle_mesh(mesh_full_file)
            mesh_dict[mesh_name] = mesh_obj

        geom_transform = {}

        body_to_mesh = defaultdict(set)
        mesh_to_body = {}
        for geom_node in tqdm(geoms):
            if "mesh" in geom_node.attrib:
                parent = find_parent(xml_doc_root, geom_node)
                body_to_mesh[parent.attrib["name"]].add(geom_node.attrib["mesh"])
                mesh_to_body[geom_node] = parent
                if "pos" in geom_node.attrib or "quat" in geom_node.attrib:
                    geom_transform[parent.attrib["name"]] = {}
                    geom_transform[parent.attrib["name"]]["pos"] = np.array(
                        [0.0, 0.0, 0.0]
                    )
                    geom_transform[parent.attrib["name"]]["quat"] = np.array(
                        [1.0, 0.0, 0.0, 0.0]
                    )
                    if "pos" in geom_node.attrib:
                        geom_transform[parent.attrib["name"]]["pos"] = np.array(
                            [float(f) for f in geom_node.attrib["pos"].split(" ")]
                        )
                    if "quat" in geom_node.attrib:
                        geom_transform[parent.attrib["name"]]["quat"] = np.array(
                            [float(f) for f in geom_node.attrib["quat"].split(" ")]
                        )

            else:
                pass

        self.geom_transform = geom_transform
        self.mesh_dict = mesh_dict
        self.body_to_mesh = body_to_mesh
        self.mesh_to_body = mesh_to_body

    def mesh_fk(self, pose=None, trans=None):
        """
        Load the mesh from the XML file and merge them into the humanoid based on the current pose.
        """
        if pose is None:
            fk_res = self.fk_batch(
                torch.zeros(1, 1, len(self.body_names_augment), 3), torch.zeros(1, 1, 3)
            )
        else:
            fk_res = self.fk_batch(pose, trans)

        g_trans = fk_res.global_translation.squeeze()
        g_rot = fk_res.global_rotation_mat.squeeze()
        geoms = self.tree.find("worldbody").findall(".//geom")
        joined_mesh_obj = []
        for geom in geoms:
            if "mesh" not in geom.attrib:
                continue
            parent_name = geom.attrib["mesh"]

            k = self.mesh_to_body[geom].attrib["name"]
            mesh_names = self.body_to_mesh[k]
            body_idx = self.body_names.index(k)

            body_trans = g_trans[body_idx].numpy().copy()
            body_rot = g_rot[body_idx].numpy().copy()
            for mesh_name in mesh_names:
                mesh_obj = copy.deepcopy(self.mesh_dict[mesh_name])
                if k in self.geom_transform:
                    pos = self.geom_transform[k]["pos"]
                    quat = self.geom_transform[k]["quat"]
                    body_trans = body_trans + body_rot @ pos
                    global_rot = (
                        body_rot @ sRot.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
                    ).T
                else:
                    global_rot = body_rot.T
                mesh_obj.rotate(global_rot.T, center=(0, 0, 0))
                mesh_obj.translate(body_trans)
                joined_mesh_obj.append(mesh_obj)

        # Merge all meshes into a single mesh
        merged_mesh = joined_mesh_obj[0]
        for mesh in joined_mesh_obj[1:]:
            merged_mesh += mesh

        # Save the merged mesh to a file
        # merged_mesh.compute_vertex_normals()
        # o3d.io.write_triangle_mesh(f"data/{self.cfg.humanoid_type}/combined_{self.cfg.humanoid_type}.stl", merged_mesh)
        return merged_mesh


@hydra.main(version_base=None, config_path="../../phc/data/cfg", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cpu")
    humanoid_fk = Humanoid_Batch(cfg.robot, device)
    humanoid_fk.mesh_fk()


if __name__ == "__main__":
    main()
