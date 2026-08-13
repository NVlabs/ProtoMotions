# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Viser-based visualizer for SMPL .motion files.

Supports both individual .motion files and packaged .pt motion libraries.
Can display two motions side-by-side for comparison (e.g., SOMA23 vs SMPL).

Usage::

    # Single SMPL motion
    python examples/visualize_smpl_motion.py --motion-file /path/to/motion.motion

    # Side-by-side comparison (SOMA23 left, SMPL right)
    python examples/visualize_smpl_motion.py \
        --motion-file /path/to/smpl.motion \
        --compare /path/to/soma23.motion --compare-robot soma23

    # Specify robot (default: auto-detect from body count)
    python examples/visualize_smpl_motion.py --motion-file /path/to/motion.motion --robot smpl
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
import trimesh.creation
import trimesh.transformations as tf
import viser

# ── Skeleton definitions ──────────────────────────────────────────────────

SKELETONS = {
    "smpl": {
        "bodies": [
            "Pelvis",
            "L_Hip",
            "L_Knee",
            "L_Ankle",
            "L_Toe",
            "R_Hip",
            "R_Knee",
            "R_Ankle",
            "R_Toe",
            "Torso",
            "Spine",
            "Chest",
            "Neck",
            "Head",
            "L_Thorax",
            "L_Shoulder",
            "L_Elbow",
            "L_Wrist",
            "L_Hand",
            "R_Thorax",
            "R_Shoulder",
            "R_Elbow",
            "R_Wrist",
            "R_Hand",
        ],
        "parents": [
            -1,
            0,
            1,
            2,
            3,
            0,
            5,
            6,
            7,
            0,
            9,
            10,
            11,
            12,
            11,
            14,
            15,
            16,
            17,
            11,
            19,
            20,
            21,
            22,
        ],
        "regions": {
            "spine": [0, 9, 10, 11, 12, 13],
            "left_arm": [14, 15, 16, 17, 18],
            "right_arm": [19, 20, 21, 22, 23],
            "left_leg": [1, 2, 3, 4],
            "right_leg": [5, 6, 7, 8],
        },
    },
    "soma23": {
        "bodies": [
            "Hips",
            "Spine1",
            "Spine2",
            "Chest",
            "Neck1",
            "Neck2",
            "Head",
            "RightShoulder",
            "RightArm",
            "RightForeArm",
            "RightHand",
            "LeftShoulder",
            "LeftArm",
            "LeftForeArm",
            "LeftHand",
            "RightLeg",
            "RightShin",
            "RightFoot",
            "RightToeBase",
            "LeftLeg",
            "LeftShin",
            "LeftFoot",
            "LeftToeBase",
        ],
        "parents": [
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            3,
            7,
            8,
            9,
            3,
            11,
            12,
            13,
            0,
            15,
            16,
            17,
            0,
            19,
            20,
            21,
        ],
        "regions": {
            "spine": [0, 1, 2, 3, 4, 5, 6],
            "right_arm": [7, 8, 9, 10],
            "left_arm": [11, 12, 13, 14],
            "right_leg": [15, 16, 17, 18],
            "left_leg": [19, 20, 21, 22],
        },
    },
}

REGION_COLORS = {
    "spine": [220, 200, 170],
    "left_arm": [240, 140, 100],
    "right_arm": [100, 170, 240],
    "left_leg": [210, 130, 210],
    "right_leg": [100, 210, 150],
}

# SMPL body geom definitions from smpl_humanoid.xml (body_idx, type, params)
# fmt: off
SMPL_GEOM_DEFS = [
    (0, "box", {"half": [0.083, 0.1069, 0.0722], "pos": [-0.0055, 0.0, -0.0121], "quat": [1, 0, 0, 0]}),
    (1, "capsule", {"radius": 0.0615, "fromto": [-0.0009, 0.0069, -0.075, -0.0036, 0.0274, -0.3002]}),
    (2, "capsule", {"radius": 0.0541, "fromto": [-0.0087, -0.0027, -0.0796, -0.035, -0.0109, -0.3184]}),
    (3, "box", {"half": [0.085, 0.0483, 0.0464], "pos": [0.0242, 0.0233, -0.0239], "quat": [1, 0, 0, 0]}),
    (4, "box", {"half": [0.0496, 0.0478, 0.02], "pos": [0.0248, -0.003, 0.0055], "quat": [1, 0, 0, 0]}),
    (5, "capsule", {"radius": 0.0606, "fromto": [-0.0018, -0.0077, -0.0765, -0.0071, -0.0306, -0.3061]}),
    (6, "capsule", {"radius": 0.0541, "fromto": [-0.0085, 0.0032, -0.0797, -0.0338, 0.0126, -0.3187]}),
    (7, "box", {"half": [0.0865, 0.0483, 0.0478], "pos": [0.0256, -0.0212, -0.0174], "quat": [1, 0, 0, 0]}),
    (8, "box", {"half": [0.0493, 0.0479, 0.0216], "pos": [0.0227, 0.0042, 0.0045], "quat": [1, 0, 0, 0]}),
    (9, "capsule", {"radius": 0.0769, "fromto": [0.0005, 0.0025, 0.0608, 0.0006, 0.003, 0.0743]}),
    (10, "capsule", {"radius": 0.0755, "fromto": [0.0114, 0.0007, 0.0238, 0.014, 0.0008, 0.0291]}),
    (11, "capsule", {"radius": 0.1002, "fromto": [-0.0173, -0.0009, 0.0682, -0.0212, -0.001, 0.0833]}),
    (12, "capsule", {"radius": 0.0436, "fromto": [0.0103, 0.001, 0.013, 0.0411, 0.0041, 0.052]}),
    (13, "box", {"half": [0.076, 0.0606, 0.1154], "pos": [-0.0116, -0.0042, 0.0876], "quat": [1, 0, 0, 0]}),
    (14, "capsule", {"radius": 0.0521, "fromto": [-0.0018, 0.0182, 0.0061, -0.0071, 0.0728, 0.0244]}),
    (15, "capsule", {"radius": 0.0517, "fromto": [-0.0055, 0.0519, -0.0026, -0.022, 0.2077, -0.0102]}),
    (16, "capsule", {"radius": 0.0405, "fromto": [-0.0002, 0.0498, 0.0018, -0.0009, 0.1994, 0.0072]}),
    (17, "capsule", {"radius": 0.0318, "fromto": [-0.003, 0.0168, -0.0016, -0.012, 0.0672, -0.0065]}),
    (18, "box", {"half": [0.0538, 0.0585, 0.0158], "pos": [-0.0058, 0.0493, 0.001], "quat": [1, 0, 0, 0]}),
    (19, "capsule", {"radius": 0.0511, "fromto": [-0.0018, -0.0192, 0.0065, -0.0073, -0.0768, 0.026]}),
    (20, "capsule", {"radius": 0.0531, "fromto": [-0.0043, -0.0507, -0.0027, -0.0171, -0.203, -0.0107]}),
    (21, "capsule", {"radius": 0.0408, "fromto": [-0.0011, -0.0511, 0.0016, -0.0044, -0.2042, 0.0062]}),
    (22, "capsule", {"radius": 0.0326, "fromto": [-0.0021, -0.0169, -0.0012, -0.0083, -0.0677, -0.0049]}),
    (23, "box", {"half": [0.0546, 0.0569, 0.0164], "pos": [-0.0079, -0.0462, -0.0009], "quat": [1, 0, 0, 0]}),
]
# fmt: on


def _build_body_meshes(geom_defs, body_regions):
    """Build trimesh for each body in its local frame from MJCF geom specs."""
    meshes = {}
    for body_idx, geom_type, params in geom_defs:
        if geom_type == "capsule":
            p0 = np.array(params["fromto"][:3])
            p1 = np.array(params["fromto"][3:])
            length = np.linalg.norm(p1 - p0)
            if length < 1e-6:
                m = trimesh.creation.icosphere(subdivisions=1, radius=params["radius"])
            else:
                m = trimesh.creation.capsule(
                    height=length, radius=params["radius"], count=[8, 8]
                )
                mid = (p0 + p1) / 2
                direction = (p1 - p0) / length
                rot_mat = _rotation_between(np.array([0, 0, 1.0]), direction)
                T = np.eye(4)
                T[:3, :3] = rot_mat
                T[:3, 3] = mid
                m.apply_transform(T)
        elif geom_type == "box":
            m = trimesh.creation.box(extents=[2 * h for h in params["half"]])
            T = np.eye(4)
            T[:3, 3] = params["pos"]
            m.apply_transform(T)
        elif geom_type == "sphere":
            m = trimesh.creation.icosphere(subdivisions=1, radius=params["radius"])
            m.apply_translation(params["pos"])
        else:
            continue
        region = body_regions.get(body_idx, "spine")
        color = REGION_COLORS[region]
        m.visual.face_colors = color + [200]
        meshes[body_idx] = m
    return meshes


def _rotation_between(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if c < -0.9999:
        perp = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
        perp = perp - np.dot(perp, a) * a
        perp /= np.linalg.norm(perp)
        return tf.rotation_matrix(np.pi, perp)[:3, :3]
    s = np.linalg.norm(v)
    if s < 1e-10:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def quat_xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


def detect_robot(num_bodies: int) -> str:
    if num_bodies == 24:
        return "smpl"
    elif num_bodies == 23:
        return "soma23"
    raise ValueError(f"Unknown robot with {num_bodies} bodies")


def load_motion(path: str) -> dict:
    """Load a .motion file (individual or packaged .pt)."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if "length_starts" in data:
        # Packaged motion lib
        motions = []
        for i in range(len(data["length_starts"])):
            start = data["length_starts"][i].item()
            n = data["motion_num_frames"][i].item()
            m = {
                "gts": data["gts"][start : start + n].numpy(),
                "grs": data["grs"][start : start + n].numpy(),
                "contacts": data["contacts"][start : start + n]
                if "contacts" in data
                else None,
                "fps": 1.0 / data["motion_dt"][i].item(),
                "name": data["motion_files"][i]
                if "motion_files" in data
                else f"motion_{i}",
            }
            motions.append(m)
        return {"motions": motions}
    elif "gts" in data and data["gts"].dim() == 3:
        # Individual .motion file: [T, B, 3]
        return {
            "motions": [
                {
                    "gts": data["gts"].numpy(),
                    "grs": data["grs"].numpy(),
                    "contacts": data.get("contacts", data.get("contacts_ground")),
                    "fps": data.get("fps", 30),
                    "name": path.split("/")[-1],
                }
            ]
        }
    raise ValueError(f"Unknown format in {path}")


def create_skeleton_meshes(skeleton_name: str):
    """Create simple capsule meshes for skeleton visualization."""
    skel = SKELETONS[skeleton_name]
    body_to_region = {}
    for region, indices in skel["regions"].items():
        for idx in indices:
            body_to_region[idx] = region
    return body_to_region


def main():
    parser = argparse.ArgumentParser(description="Viser SMPL/SOMA23 motion visualizer")
    parser.add_argument(
        "--motion-file", required=True, help="Path to .motion or .pt file"
    )
    parser.add_argument(
        "--robot",
        default=None,
        help="Robot type (smpl/soma23). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--compare",
        default=None,
        help="Second .motion file for side-by-side comparison",
    )
    parser.add_argument(
        "--compare-robot", default=None, help="Robot type for comparison motion"
    )
    parser.add_argument(
        "--scene-mesh",
        default=None,
        nargs="*",
        help="One or more .obj mesh files to display as static scene objects",
    )
    parser.add_argument(
        "--scene-dir",
        default=None,
        help="Directory with per-motion scene .obj files (auto-loaded by motion name)",
    )
    parser.add_argument(
        "--scene-pt",
        default=None,
        help="Packaged scene .pt file (indexed by motion slider, matching motion .pt)",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--offset", type=float, default=1.5, help="X offset between compared motions"
    )
    args = parser.parse_args()

    # Load primary motion
    motion_data = load_motion(args.motion_file)
    motions = motion_data["motions"]
    num_bodies = motions[0]["gts"].shape[1]
    robot = args.robot or detect_robot(num_bodies)
    skel = SKELETONS[robot]
    body_to_region = create_skeleton_meshes(robot)

    # Load comparison motion
    cmp_motions = None
    cmp_skel = None
    cmp_body_to_region = None
    if args.compare:
        cmp_data = load_motion(args.compare)
        cmp_motions = cmp_data["motions"]
        cmp_num_bodies = cmp_motions[0]["gts"].shape[1]
        cmp_robot = args.compare_robot or detect_robot(cmp_num_bodies)
        cmp_skel = SKELETONS[cmp_robot]
        cmp_body_to_region = create_skeleton_meshes(cmp_robot)

    # Load scene meshes
    scene_meshes = []
    if args.scene_mesh:
        for mesh_path in args.scene_mesh:
            try:
                mesh = trimesh.load(mesh_path, force="mesh")
                scene_meshes.append((mesh_path.split("/")[-1], mesh))
                print(f"Loaded scene mesh: {mesh_path} ({len(mesh.vertices)} verts)")
            except Exception as e:
                print(f"Warning: failed to load {mesh_path}: {e}")

    server = viser.ViserServer(port=args.port)
    print(f"Viser running at http://localhost:{args.port}")

    # GUI controls
    motion_idx_slider = server.gui.add_slider(
        "Motion", min=0, max=len(motions) - 1, step=1, initial_value=0
    )
    frame_slider = server.gui.add_slider(
        "Frame", min=0, max=motions[0]["gts"].shape[0] - 1, step=1, initial_value=0
    )
    playing = server.gui.add_checkbox("Play", initial_value=False)
    speed = server.gui.add_slider(
        "Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0
    )
    _sphere_radius = server.gui.add_slider(
        "Body radius", min=0.01, max=0.08, step=0.005, initial_value=0.04
    )

    # Add static scene meshes (from --scene-mesh)
    for i, (name, mesh) in enumerate(scene_meshes):
        if not hasattr(mesh.visual, "face_colors") or mesh.visual.face_colors is None:
            mesh.visual.face_colors = [180, 180, 180, 200]
        server.scene.add_mesh_trimesh(f"/scene/{name}", mesh)

    # Add ground plane
    ground = trimesh.creation.box(extents=[20, 20, 0.005])
    ground.visual.face_colors = [200, 200, 200, 100]
    ground.apply_translation([0, 0, -0.0025])
    server.scene.add_mesh_trimesh("/scene/ground", ground)

    # Per-motion scene loading
    scene_dir = Path(args.scene_dir) if args.scene_dir else None
    # Load packaged scene .pt if provided
    packaged_scenes = None
    if args.scene_pt:
        packaged_scenes = torch.load(
            args.scene_pt, weights_only=False, map_location="cpu"
        )
        print(
            f"Loaded packaged scenes: {packaged_scenes['num_original_scenes']} scenes, "
            f"{packaged_scenes['num_objects_per_scene']} objects/scene"
        )

    current_scene_handles = []
    current_scene_idx = [-1]

    def _clear_scene():
        for h in current_scene_handles:
            h.remove()
        current_scene_handles.clear()

    def _render_scene_objects(scene: dict):
        """Render BoxSceneObjects from a scene dict."""
        for j, obj in enumerate(scene.get("objects", [])):
            dims = obj.get("object_dims", None)
            if dims is None:
                continue
            min_x, max_x, min_y, max_y, min_z, max_z = dims
            # Skip dummy objects (buried underground)
            if max_z < -5:
                continue
            extents = np.array([max_x - min_x, max_y - min_y, max_z - min_z])
            center = np.array(
                [(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2]
            )
            trans = obj.get("translation", [[0, 0, 0]])[0]
            center = center + np.array(trans)
            rot_xyzw = obj.get("rotation", [[0, 0, 0, 1]])[0]
            wxyz = quat_xyzw_to_wxyz(np.array(rot_xyzw))
            box = trimesh.creation.box(extents=extents)
            box.visual.face_colors = [160, 160, 180, 120]
            h = server.scene.add_mesh_trimesh(
                f"/scene/obj_{j}", box, position=center, wxyz=wxyz
            )
            current_scene_handles.append(h)

    def load_scene_for_motion(mi: int):
        """Load/swap scene for the current motion."""
        if mi == current_scene_idx[0]:
            return
        _clear_scene()
        current_scene_idx[0] = mi

        # Packaged scene .pt — index directly by motion index
        if packaged_scenes is not None:
            scenes = packaged_scenes["original_scenes"]
            if mi < len(scenes):
                _render_scene_objects(scenes[mi])
                n_real = sum(
                    1
                    for o in scenes[mi]["objects"]
                    if o.get("object_dims", (0, 0, 0, 0, 0, -10))[5] > -5
                )
                print(f"Scene {mi}: {n_real} objects")
            return

        # Per-file scene loading from --scene-dir
        if scene_dir is None or not scene_dir.exists():
            return

        motion_name = motions[mi].get("name", f"motion_{mi}")
        base = re.sub(r"(_[abcd])?(\.(motion|pkl))?$", "", motion_name)

        # Per-file .pt scene
        pt_path = scene_dir / f"{base}.pt"
        if pt_path.exists():
            try:
                sd = torch.load(str(pt_path), weights_only=False, map_location="cpu")
                if "original_scenes" in sd and sd["original_scenes"]:
                    _render_scene_objects(sd["original_scenes"][0])
            except Exception as e:
                print(f"Warning: failed to load {pt_path}: {e}")

    # Track handles for cleanup
    body_handles = {}
    bone_handles = {}
    cmp_body_handles = {}
    cmp_bone_handles = {}

    def update_frame_range():
        mi = motion_idx_slider.value
        n_frames = motions[mi]["gts"].shape[0]
        frame_slider.max = n_frames - 1
        if frame_slider.value >= n_frames:
            frame_slider.value = 0
        load_scene_for_motion(mi)

    @motion_idx_slider.on_update
    def _(_):
        update_frame_range()

    # Build body-local meshes for SMPL
    smpl_body_meshes = _build_body_meshes(SMPL_GEOM_DEFS, body_to_region)

    def render_skeleton(
        gts,
        grs,
        contacts,
        skel_info,
        body_reg,
        handles_body,
        handles_bone,
        prefix="",
        x_offset=0.0,
    ):
        """Render one skeleton frame using body-local capsule/box meshes."""
        num_b = gts.shape[0]

        for i in range(num_b):
            pos = gts[i].copy()
            pos[0] += x_offset
            rot_xyzw = grs[i]
            wxyz = quat_xyzw_to_wxyz(rot_xyzw)
            name = f"{prefix}{skel_info['bodies'][i]}"

            # Get body mesh (capsule/box from MJCF)
            body_mesh = smpl_body_meshes.get(i)
            if body_mesh is None:
                continue

            # Apply contact color
            mesh_copy = body_mesh.copy()
            if contacts is not None and contacts[i]:
                mesh_copy.visual.face_colors = [255, 50, 50, 220]

            # Recreate each frame for contact color updates
            if name in handles_body:
                handles_body[name].remove()
            handles_body[name] = server.scene.add_mesh_trimesh(
                f"/skel/{name}",
                mesh_copy,
                position=pos,
                wxyz=wxyz,
            )

    # Load initial scene
    load_scene_for_motion(0)

    # Main loop
    prev_time = time.time()
    frame_accum = 0.0

    while True:
        mi = motion_idx_slider.value
        load_scene_for_motion(mi)
        fi = frame_slider.value
        motion = motions[mi]

        gts = motion["gts"][fi]
        grs = motion["grs"][fi]
        contacts = motion["contacts"][fi] if motion["contacts"] is not None else None
        if contacts is not None:
            contacts = (
                contacts.numpy() if isinstance(contacts, torch.Tensor) else contacts
            )

        render_skeleton(
            gts, grs, contacts, skel, body_to_region, body_handles, bone_handles
        )

        if cmp_motions is not None:
            cmp_mi = min(mi, len(cmp_motions) - 1)
            cmp_fi = min(fi, cmp_motions[cmp_mi]["gts"].shape[0] - 1)
            cmp = cmp_motions[cmp_mi]
            cmp_contacts = (
                cmp["contacts"][cmp_fi] if cmp["contacts"] is not None else None
            )
            if cmp_contacts is not None:
                cmp_contacts = (
                    cmp_contacts.numpy()
                    if isinstance(cmp_contacts, torch.Tensor)
                    else cmp_contacts
                )
            render_skeleton(
                cmp["gts"][cmp_fi],
                cmp["grs"][cmp_fi],
                cmp_contacts,
                cmp_skel,
                cmp_body_to_region,
                cmp_body_handles,
                cmp_bone_handles,
                prefix="cmp_",
                x_offset=args.offset,
            )

        # Playback
        now = time.time()
        dt = now - prev_time
        prev_time = now

        if playing.value:
            fps = motion["fps"] if isinstance(motion["fps"], (int, float)) else 30
            frame_accum += dt * fps * speed.value
            if frame_accum >= 1.0:
                advance = int(frame_accum)
                frame_accum -= advance
                new_frame = fi + advance
                max_frame = motion["gts"].shape[0] - 1
                if new_frame > max_frame:
                    new_frame = 0
                frame_slider.value = new_frame

        time.sleep(0.016)  # ~60 fps render


if __name__ == "__main__":
    main()
