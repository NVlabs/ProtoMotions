# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
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

"""
Optimize Blender mesh skinning weights to match MJCF rigid body attachment.

In MJCF, each STL visual mesh rigidly follows its parent body. In Blender,
a single joined mesh is deformed by an armature via Linear Blend Skinning (LBS):

    vertex_pos = Σ_b weight[v,b] × BoneTransform[b] × rest_pos[v]

This script optimizes the weights so the Blender deformation matches the MJCF
rigid attachment as closely as possible. For a robot (no stretchy skin), this
typically means assigning each vertex to exactly one bone (weight=1.0).

Approach:
  1. Parse MJCF to get per-body STL mesh files and rest-pose body transforms
  2. Load STL vertices per body, transform to world space (rest pose)
  3. Match each Blender mesh vertex to the nearest STL vertex → body ownership
  4. Assign vertex weights: weight=1.0 to the owning body's bone

Usage:
    python scripts/optimize_blend_weights.py \\
        --blend-file protomotions/data/assets/blender/g1_asset.blend \\
        --output-file protomotions/data/assets/blender/g1_asset_optimized.blend

    # Verify with a render:
    python blender/render_motion_blender.py --robot g1 --motion-file ... \\
        --film-strip 15 --mesh-overlay --output-dir renders/verify
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

try:
    import bpy  # noqa: F401
    import mathutils  # noqa: F401

    HAS_BPY = True
except ImportError:
    HAS_BPY = False


# ---------------------------------------------------------------------------
# Robot config (same as render script — bone name mapping)
# ---------------------------------------------------------------------------
G1_CONFIG = {
    "mjcf": "protomotions/data/assets/mjcf/g1_holo_compat.xml",
    "blend_asset": "protomotions/data/assets/blender/g1_asset.blend",
    "armature_name": "g1",
    "mesh_z_offset": 0.8,
    "bvh_bone_names": [
        "root",
        "head",
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "left_rubber_hand",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
        "right_rubber_hand",
    ],
}


# ---------------------------------------------------------------------------
# MJCF parsing (self-contained, no torch dependency)
# ---------------------------------------------------------------------------


@dataclass
class VisualMeshInfo:
    mesh_name: str
    stl_path: str
    geom_pos: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    geom_quat: list = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])  # wxyz


@dataclass
class BodyVisualInfo:
    name: str
    parent_idx: int
    pos: list  # local position relative to parent
    quat: list  # local orientation relative to parent (wxyz)
    meshes: list
    has_joints: bool


def parse_mjcf_visual(mjcf_path: str) -> list[BodyVisualInfo]:
    """Parse MJCF and return per-body visual mesh info with body rest transforms."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    meshdir = ""
    compiler = root.find("compiler")
    if compiler is not None:
        meshdir = compiler.attrib.get("meshdir", "")

    mjcf_dir = os.path.dirname(os.path.abspath(mjcf_path))
    mesh_dir = (
        os.path.normpath(os.path.join(mjcf_dir, meshdir)) if meshdir else mjcf_dir
    )

    mesh_name_to_file = {}
    asset = root.find("asset")
    if asset is not None:
        for mesh_el in asset.findall("mesh"):
            name = mesh_el.attrib.get("name", "")
            fname = mesh_el.attrib.get("file", "")
            if name and fname:
                mesh_name_to_file[name] = os.path.join(mesh_dir, fname)

    worldbody = root.find("worldbody")
    xml_body_root = worldbody.find("body")
    bodies = []

    def _parse_visual_meshes(xml_node):
        seen = set()
        meshes = []
        for geom_node in xml_node.findall("geom"):
            gtype = geom_node.attrib.get("type", "")
            mesh_name = geom_node.attrib.get("mesh", "")
            if gtype == "mesh" and mesh_name and mesh_name not in seen:
                stl_path = mesh_name_to_file.get(mesh_name, "")
                if stl_path and os.path.exists(stl_path):
                    gpos = [
                        float(v) for v in geom_node.attrib.get("pos", "0 0 0").split()
                    ]
                    gquat = [
                        float(v)
                        for v in geom_node.attrib.get("quat", "1 0 0 0").split()
                    ]
                    meshes.append(
                        VisualMeshInfo(
                            mesh_name=mesh_name,
                            stl_path=stl_path,
                            geom_pos=gpos,
                            geom_quat=gquat,
                        )
                    )
                    seen.add(mesh_name)
        return meshes

    def _add_body(xml_node, parent_index, body_index):
        name = xml_node.attrib.get("name", f"body_{body_index}")
        pos = [float(v) for v in xml_node.attrib.get("pos", "0 0 0").split()]
        quat = [float(v) for v in xml_node.attrib.get("quat", "1 0 0 0").split()]
        meshes = _parse_visual_meshes(xml_node)
        has_joints = len(xml_node.findall("joint")) > 0
        bodies.append(
            BodyVisualInfo(
                name=name,
                parent_idx=parent_index,
                pos=pos,
                quat=quat,
                meshes=meshes,
                has_joints=has_joints,
            )
        )
        curr_index = body_index
        body_index += 1
        for child in xml_node.findall("body"):
            body_index = _add_body(child, curr_index, body_index)
        return body_index

    _add_body(xml_body_root, -1, 0)
    return bodies


# ---------------------------------------------------------------------------
# Quaternion math (numpy, no torch)
# ---------------------------------------------------------------------------


def quat_mul(q1, q2):
    """Multiply two quaternions (wxyz convention). Supports batched."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_rotate(q, v):
    """Rotate vector v by quaternion q (wxyz). Returns rotated vector."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    # q * (0,v) * q_conj
    t = 2.0 * np.cross(np.array([x, y, z]), v)
    return v + w * t + np.cross(np.array([x, y, z]), t)


def load_stl_vertices(stl_path: str) -> np.ndarray:
    """Load unique vertices from a binary STL file. Returns (N, 3)."""
    with open(stl_path, "rb") as f:
        f.read(80)  # header
        n_triangles = int.from_bytes(f.read(4), "little")
        vertices = []
        for _ in range(n_triangles):
            f.read(12)  # normal
            for _ in range(3):
                vx = np.frombuffer(f.read(4), dtype=np.float32)[0]
                vy = np.frombuffer(f.read(4), dtype=np.float32)[0]
                vz = np.frombuffer(f.read(4), dtype=np.float32)[0]
                vertices.append((vx, vy, vz))
            f.read(2)  # attribute byte count

    verts = np.array(vertices, dtype=np.float64)
    # Deduplicate
    unique_verts = np.unique(np.round(verts, decimals=6), axis=0)
    return unique_verts


def compute_body_world_transforms(bodies):
    """Compute rest-pose world position and orientation for each body.

    Returns (positions, orientations) as lists of numpy arrays.
    positions[i] is (3,), orientations[i] is (4,) wxyz.
    """
    n = len(bodies)
    world_pos = [None] * n
    world_quat = [None] * n

    for i, body in enumerate(bodies):
        local_pos = np.array(body.pos)
        local_quat = np.array(body.quat)

        if body.parent_idx < 0:
            world_pos[i] = local_pos
            world_quat[i] = local_quat
        else:
            parent_pos = world_pos[body.parent_idx]
            parent_quat = world_quat[body.parent_idx]
            # World pos = parent_pos + parent_quat.rotate(local_pos)
            world_pos[i] = parent_pos + quat_rotate(parent_quat, local_pos)
            world_quat[i] = quat_mul(parent_quat, local_quat)

    return world_pos, world_quat


def build_stl_world_vertices(bodies, world_pos, world_quat):
    """For each body, load STL vertices and transform to world space.

    Returns list of (body_idx, world_vertices_Nx3) tuples.
    """
    result = []
    for b_idx, body in enumerate(bodies):
        for mesh_info in body.meshes:
            local_verts = load_stl_vertices(mesh_info.stl_path)
            if local_verts.shape[0] == 0:
                continue

            # Apply geom-level offset (pos/quat) first
            gp = np.array(mesh_info.geom_pos)
            gq = np.array(mesh_info.geom_quat)
            if not (np.allclose(gp, 0) and np.allclose(gq, [1, 0, 0, 0])):
                transformed = np.array([quat_rotate(gq, v) + gp for v in local_verts])
                local_verts = transformed

            # Transform to world space
            bp = world_pos[b_idx]
            bq = world_quat[b_idx]
            world_verts = np.array([quat_rotate(bq, v) + bp for v in local_verts])
            result.append((b_idx, world_verts))

    return result


# ---------------------------------------------------------------------------
# Blender weight assignment (runs inside Blender)
# ---------------------------------------------------------------------------


def optimize_weights_in_blender(
    blend_path: str,
    mjcf_path: str,
    output_path: str,
    armature_name: str,
    bone_names: list[str],
    mesh_z_offset: float,
):
    """Open blend file, reassign vertex weights from MJCF body ownership."""
    import bpy

    # Open the blend file
    bpy.ops.wm.open_mainfile(filepath=blend_path)

    # Find armature and rigged mesh
    armature = None
    mesh_obj = None
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE" and obj.name == armature_name:
            armature = obj
        elif obj.type == "MESH" and len(obj.vertex_groups) > 0:
            mesh_obj = obj

    if armature is None:
        print(f"ERROR: Armature '{armature_name}' not found")
        return
    if mesh_obj is None:
        print("ERROR: No rigged mesh found")
        return

    print(f"Armature: {armature.name}, Mesh: {mesh_obj.name}")
    print(f"Vertices: {len(mesh_obj.data.vertices)}")
    print(f"Existing vertex groups: {[vg.name for vg in mesh_obj.vertex_groups]}")

    # Parse MJCF and compute body world transforms
    bodies = parse_mjcf_visual(mjcf_path)
    world_pos, world_quat = compute_body_world_transforms(bodies)

    print(f"\nMJCF bodies: {len(bodies)}")
    for i, b in enumerate(bodies):
        mesh_names = [m.mesh_name for m in b.meshes]
        bone = bone_names[i] if i < len(bone_names) else "???"
        print(f"  [{i}] {b.name} -> bone '{bone}', meshes: {mesh_names}")

    # Build world-space STL vertex cloud per body
    body_stl_data = build_stl_world_vertices(bodies, world_pos, world_quat)

    # Combine all STL vertices with body labels for KD-tree lookup
    all_stl_verts = []
    all_stl_body_idx = []
    for b_idx, verts in body_stl_data:
        all_stl_verts.append(verts)
        all_stl_body_idx.extend([b_idx] * verts.shape[0])

    all_stl_verts = np.vstack(all_stl_verts)
    all_stl_body_idx = np.array(all_stl_body_idx)
    print(f"\nTotal STL vertices (world space): {all_stl_verts.shape[0]}")

    # Get Blender mesh vertices in world space
    # The mesh object has a transform (location at z=mesh_z_offset)
    mesh_world_matrix = mesh_obj.matrix_world
    blender_verts = np.array(
        [(mesh_world_matrix @ v.co).to_tuple() for v in mesh_obj.data.vertices]
    )
    print(f"Blender mesh vertices: {blender_verts.shape[0]}")

    # Build KD-tree from STL vertices for fast nearest-neighbor lookup
    from scipy.spatial import cKDTree

    tree = cKDTree(all_stl_verts)
    distances, indices = tree.query(blender_verts, k=1)

    # Map each Blender vertex to its nearest STL body
    vertex_body_map = all_stl_body_idx[indices]

    # Report statistics
    max_dist = distances.max()
    mean_dist = distances.mean()
    median_dist = np.median(distances)
    print("\nVertex matching distances:")
    print(f"  Max:    {max_dist:.6f} m")
    print(f"  Mean:   {mean_dist:.6f} m")
    print(f"  Median: {median_dist:.6f} m")

    # Count per body
    unique_bodies, counts = np.unique(vertex_body_map, return_counts=True)
    print("\nVertex ownership:")
    for b_idx, count in zip(unique_bodies, counts):
        bone = bone_names[b_idx] if b_idx < len(bone_names) else "???"
        print(f"  [{b_idx}] {bodies[b_idx].name} -> bone '{bone}': {count} vertices")

    # Flag vertices with large distances (potential issues)
    threshold = 0.01  # 1cm
    far_verts = distances > threshold
    if far_verts.any():
        print(
            f"\nWARNING: {far_verts.sum()} vertices are >{threshold*100:.0f}cm "
            f"from nearest STL vertex (max={max_dist:.4f}m)"
        )

    # Clear existing vertex groups and create new ones
    mesh_obj.vertex_groups.clear()
    bone_to_vg = {}
    for i, bone_name in enumerate(bone_names):
        if i < len(bodies):
            vg = mesh_obj.vertex_groups.new(name=bone_name)
            bone_to_vg[i] = vg

    # Assign weights: 1.0 to the owning bone for each vertex
    for v_idx in range(len(mesh_obj.data.vertices)):
        b_idx = vertex_body_map[v_idx]
        if b_idx in bone_to_vg:
            bone_to_vg[b_idx].add([v_idx], 1.0, "REPLACE")

    print(f"\nAssigned rigid weights to {len(mesh_obj.data.vertices)} vertices")

    # Verify armature modifier is set up
    has_armature_mod = False
    for mod in mesh_obj.modifiers:
        if mod.type == "ARMATURE":
            mod.object = armature
            has_armature_mod = True
    if not has_armature_mod:
        mod = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
        mod.object = armature
        print("Added Armature modifier")

    # Save
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(output_path))
    print(f"\nSaved optimized blend file: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def find_blender():
    candidates = [
        os.path.expanduser("~/Downloads/blender"),
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "blender",
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return shutil.which("blender") or candidates[0]


def main():
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    else:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Optimize Blender mesh skinning weights to match MJCF rigid attachment"
    )
    parser.add_argument(
        "--blend-file",
        default=str(_PROJECT_ROOT / G1_CONFIG["blend_asset"]),
        help="Path to input .blend file",
    )
    parser.add_argument(
        "--output-file",
        default=str(
            _PROJECT_ROOT / "protomotions/data/assets/blender/g1_asset_optimized.blend"
        ),
        help="Path to output .blend file",
    )
    parser.add_argument(
        "--mjcf",
        default=str(_PROJECT_ROOT / G1_CONFIG["mjcf"]),
        help="Path to MJCF XML file",
    )

    args = parser.parse_args(argv)

    if not HAS_BPY:
        # Relaunch inside Blender
        blender_path = find_blender()
        cmd = [
            blender_path,
            "--background",
            "--python",
            str(Path(__file__).resolve()),
            "--",
        ] + argv
        print(f"Launching Blender: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return

    # Running inside Blender
    optimize_weights_in_blender(
        blend_path=args.blend_file,
        mjcf_path=args.mjcf,
        output_path=args.output_file,
        armature_name=G1_CONFIG["armature_name"],
        bone_names=G1_CONFIG["bvh_bone_names"],
        mesh_z_offset=G1_CONFIG["mesh_z_offset"],
    )


if __name__ == "__main__":
    main()
