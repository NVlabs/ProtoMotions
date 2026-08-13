# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Viser-based visualizer for SOMA23 packaged motions + scene objects.

Usage:
    python examples/visualize_soma_scenes.py \
        --motion-file /path/to/motions.pt \
        --scene-file /path/to/scenes.pt \
        --mesh-root /path/to/meshes/   # directory containing .obj files
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
import trimesh.creation
import trimesh.transformations as tf
import viser

# -- SOMA23 skeleton definition (from soma23_humanoid.xml) --
BODY_NAMES = [
    "Hips",  # 0
    "Spine1",  # 1
    "Spine2",  # 2
    "Chest",  # 3
    "Neck1",  # 4
    "Neck2",  # 5
    "Head",  # 6
    "RightShoulder",  # 7
    "RightArm",  # 8
    "RightForeArm",  # 9
    "RightHand",  # 10
    "LeftShoulder",  # 11
    "LeftArm",  # 12
    "LeftForeArm",  # 13
    "LeftHand",  # 14
    "RightLeg",  # 15
    "RightShin",  # 16
    "RightFoot",  # 17
    "RightToeBase",  # 18
    "LeftLeg",  # 19
    "LeftShin",  # 20
    "LeftFoot",  # 21
    "LeftToeBase",  # 22
]

PARENT_INDICES = [
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
]
BONES = [
    (PARENT_INDICES[i], i) for i in range(len(BODY_NAMES)) if PARENT_INDICES[i] >= 0
]

# Body region colors (R, G, B) 0-255
REGION_COLORS = {
    "spine": [220, 200, 170],
    "right_arm": [100, 170, 240],
    "left_arm": [240, 140, 100],
    "right_leg": [100, 210, 150],
    "left_leg": [210, 130, 210],
}
BODY_REGION = {}
for i in [0, 1, 2, 3, 4, 5, 6]:
    BODY_REGION[i] = "spine"
for i in [7, 8, 9, 10]:
    BODY_REGION[i] = "right_arm"
for i in [11, 12, 13, 14]:
    BODY_REGION[i] = "left_arm"
for i in [15, 16, 17, 18]:
    BODY_REGION[i] = "right_leg"
for i in [19, 20, 21, 22]:
    BODY_REGION[i] = "left_leg"


def make_capsule_mesh(radius, fromto):
    """Create a trimesh capsule from MuJoCo fromto spec, in body-local frame."""
    p0 = np.array(fromto[:3])
    p1 = np.array(fromto[3:])
    length = np.linalg.norm(p1 - p0)
    if length < 1e-6:
        return trimesh.creation.icosphere(subdivisions=1, radius=radius)
    # trimesh capsule is along Z axis, centered at origin
    cap = trimesh.creation.capsule(height=length, radius=radius, count=[8, 8])
    # We need to transform: capsule default is along Z, from -h/2 to +h/2
    # Target: from p0 to p1
    mid = (p0 + p1) / 2.0
    direction = (p1 - p0) / length
    # Rotation from Z to direction
    z_axis = np.array([0, 0, 1.0])
    rot_mat = _rotation_between(z_axis, direction)
    T = np.eye(4)
    T[:3, :3] = rot_mat
    T[:3, 3] = mid
    cap.apply_transform(T)
    return cap


def _rotation_between(a, b):
    """Rotation matrix from unit vector a to unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if c < -0.9999:
        # Nearly opposite - pick an arbitrary perpendicular axis
        perp = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
        perp = perp - np.dot(perp, a) * a
        perp = perp / np.linalg.norm(perp)
        return tf.rotation_matrix(np.pi, perp)[:3, :3]
    s = np.linalg.norm(v)
    if s < 1e-10:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def make_box_mesh(half_extents, pos, quat_wxyz):
    """Create a trimesh box from MuJoCo spec (half extents), in body-local frame."""
    box = trimesh.creation.box(extents=[2 * h for h in half_extents])
    T = np.eye(4)
    T[:3, :3] = tf.quaternion_matrix(quat_wxyz)[:3, :3]
    T[:3, 3] = pos
    box.apply_transform(T)
    return box


def make_sphere_mesh(radius, pos):
    """Create a trimesh sphere at a local offset."""
    sphere = trimesh.creation.icosphere(subdivisions=1, radius=radius)
    sphere.apply_translation(pos)
    return sphere


def build_body_local_meshes():
    """Build a trimesh for each SOMA23 body in its local frame, from the MJCF geom specs."""
    # Geom definitions from soma23_humanoid.xml
    # Format: (body_idx, type, params)
    geom_defs = [
        (0, "sphere", {"radius": 0.08, "pos": [0, -0.03, 0]}),
        (1, "sphere", {"radius": 0.06, "pos": [0, -0.03, 0.05]}),
        (2, "sphere", {"radius": 0.07, "pos": [0, -0.04, 0.06]}),
        (3, "sphere", {"radius": 0.11, "pos": [0, -0.04, 0.12]}),
        (4, "capsule", {"radius": 0.04, "fromto": [0, -0.03, 0, 0, -0.03, 0.07]}),
        (5, "capsule", {"radius": 0.04, "fromto": [0, 0, 0, 0, -0.01, 0.05]}),
        (
            6,
            "box",
            {
                "half": [0.075, 0.075, 0.105],
                "pos": [0, 0.02, 0.07],
                "quat": [1, 0, 0, 0],
            },
        ),
        (7, "capsule", {"radius": 0.045, "fromto": [-0.045, 0.03, 0, -0.11, 0.03, 0]}),
        (8, "capsule", {"radius": 0.045, "fromto": [-0.045, 0, 0, -0.24, 0, 0]}),
        (9, "capsule", {"radius": 0.035, "fromto": [-0.04, 0, 0, -0.23, 0, 0]}),
        (
            10,
            "capsule",
            {"radius": 0.05, "fromto": [-0.02, 0, 0, -0.0807, -0.0215, -0.0158]},
        ),
        (11, "capsule", {"radius": 0.045, "fromto": [0.045, 0.03, 0, 0.11, 0.03, 0]}),
        (12, "capsule", {"radius": 0.045, "fromto": [0.045, 0, 0, 0.24, 0, 0]}),
        (13, "capsule", {"radius": 0.035, "fromto": [0.04, 0, 0, 0.23, 0, 0]}),
        (
            14,
            "capsule",
            {"radius": 0.05, "fromto": [0.02, 0, 0, 0.0808, -0.0216, -0.0159]},
        ),
        (15, "capsule", {"radius": 0.06, "fromto": [0, 0, 0, 0, 0, -0.37]}),
        (16, "capsule", {"radius": 0.05, "fromto": [0, 0, -0.05, 0, 0, -0.37]}),
        (
            17,
            "box",
            {
                "half": [0.075, 0.045, 0.028],
                "pos": [0, -0.045, -0.017],
                "quat": [0.7071, 0, 0, 0.7071],
            },
        ),
        (
            18,
            "box",
            {
                "half": [0.050, 0.032, 0.0215],
                "pos": [0, -0.012, 0.027],
                "quat": [1, 0, 0, 0],
            },
        ),
        (19, "capsule", {"radius": 0.06, "fromto": [0, 0, 0, 0, 0, -0.37]}),
        (20, "capsule", {"radius": 0.05, "fromto": [0, 0, -0.05, 0, 0, -0.37]}),
        (
            21,
            "box",
            {
                "half": [0.075, 0.045, 0.028],
                "pos": [0, -0.045, -0.017],
                "quat": [0.7071, 0, 0, 0.7071],
            },
        ),
        (
            22,
            "box",
            {
                "half": [0.050, 0.032, 0.0215],
                "pos": [0, -0.012, 0.027],
                "quat": [1, 0, 0, 0],
            },
        ),
    ]

    meshes = {}
    for body_idx, geom_type, params in geom_defs:
        if geom_type == "sphere":
            m = make_sphere_mesh(params["radius"], params["pos"])
        elif geom_type == "capsule":
            m = make_capsule_mesh(params["radius"], params["fromto"])
        elif geom_type == "box":
            m = make_box_mesh(params["half"], params["pos"], params["quat"])
        else:
            continue
        # Apply region color
        color = REGION_COLORS[BODY_REGION[body_idx]]
        m.visual.face_colors = color + [255]
        meshes[body_idx] = m
    return meshes


def quat_xyzw_to_wxyz(q):
    """Convert quaternion from xyzw to wxyz."""
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


def load_motion_data(motion_file: str) -> dict:
    data = torch.load(motion_file, map_location="cpu", weights_only=False)
    return data


def load_scene_data(scene_file: str) -> dict:
    data = torch.load(scene_file, map_location="cpu", weights_only=False)
    return data


def get_motion_frames(data: dict, motion_idx: int):
    """Extract gts, grs, contacts for a specific motion."""
    start = data["length_starts"][motion_idx].item()
    n_frames = data["motion_num_frames"][motion_idx].item()
    dt = data["motion_dt"][motion_idx].item()
    gts = data["gts"][start : start + n_frames].numpy()  # (F, 23, 3)
    grs = data["grs"][start : start + n_frames].numpy()  # (F, 23, 4) xyzw
    contacts = (
        data["contacts"][start : start + n_frames].numpy()
        if "contacts" in data
        else None
    )
    return gts, grs, contacts, dt, n_frames


def load_scene_meshes(scene_data: dict, motion_idx: int, mesh_root: Path):
    """Load scene objects as trimesh meshes (boxes or .obj files).

    Returns a list of (mesh, translation, rotation_xyzw) tuples.
    """
    if motion_idx >= len(scene_data["original_scenes"]):
        return []
    scene = scene_data["original_scenes"][motion_idx]
    if not scene["objects"]:
        return []

    results = []
    for obj in scene["objects"]:
        trans = np.array(obj["translation"][0], dtype=np.float32)
        rot = np.array(obj["rotation"][0], dtype=np.float32)

        if obj["type"] == "BoxSceneObject":
            mesh = trimesh.creation.box(
                extents=[obj["width"], obj["depth"], obj["height"]]
            )
            results.append((mesh, trans, rot))
        elif obj["type"] == "MeshSceneObject" and mesh_root is not None:
            obj_name = Path(obj["object_path"]).stem + ".obj"
            mesh_path = mesh_root / obj_name
            if mesh_path.exists():
                mesh = trimesh.load(str(mesh_path), force="mesh")
                results.append((mesh, trans, rot))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Visualize SOMA23 motions + scenes with viser"
    )
    parser.add_argument(
        "--motion-file", type=str, required=True, help="Packaged motion file (.pt)"
    )
    parser.add_argument("--scene-file", type=str, default=None, help="Scene file (.pt)")
    parser.add_argument(
        "--mesh-root",
        type=str,
        default=None,
        help="Directory containing scene .obj mesh files",
    )
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    args = parser.parse_args()

    print("Loading motion data...")
    motion_data = load_motion_data(args.motion_file)
    n_motions = len(motion_data["motion_num_frames"])
    print(f"  {n_motions} motions, {motion_data['gts'].shape[0]} total frames")

    scene_data = None
    if args.scene_file:
        print("Loading scene data...")
        scene_data = load_scene_data(args.scene_file)
        print(f"  {scene_data['num_original_scenes']} scenes")

    mesh_root = Path(args.mesh_root) if args.mesh_root else None

    print("Building body meshes from MJCF...")
    body_local_meshes = build_body_local_meshes()

    motion_names = []
    for i, f in enumerate(motion_data["motion_files"]):
        name = Path(f).stem if isinstance(f, str) else f"motion_{i}"
        motion_names.append(f"{i}: {name}")

    server = viser.ViserServer(port=args.port)
    print(f"Viser running at http://localhost:{args.port}")

    server.scene.add_grid("/grid", width=20, height=20, cell_size=1.0)

    # -- GUI --
    with server.gui.add_folder("Motion"):
        motion_slider = server.gui.add_slider(
            "Motion index", min=0, max=n_motions - 1, step=1, initial_value=0
        )
        motion_label = server.gui.add_text(
            "Name", initial_value=motion_names[0], disabled=True
        )
        frame_slider = server.gui.add_slider(
            "Frame", min=0, max=100, step=1, initial_value=0
        )
        play_btn = server.gui.add_button("Play / Pause")
        speed_slider = server.gui.add_slider(
            "Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0
        )
        show_contacts = server.gui.add_checkbox("Show contacts", initial_value=True)

    with server.gui.add_folder("Scene"):
        show_scene = server.gui.add_checkbox("Show scene mesh", initial_value=True)

    # -- State --
    playing = False
    current_motion_idx = 0
    current_gts = None
    current_grs = None
    current_contacts = None
    current_dt = 0.033
    current_n_frames = 0
    scene_mesh_handle = None

    # Create per-body viser handles (one frame + mesh per body)
    body_handles = {}
    for body_idx, local_mesh in body_local_meshes.items():
        frame_name = f"/skeleton/{BODY_NAMES[body_idx]}"
        frame_handle = server.scene.add_frame(frame_name, show_axes=False)
        mesh_handle = server.scene.add_mesh_trimesh(
            f"{frame_name}/geom", mesh=local_mesh
        )
        body_handles[body_idx] = (frame_handle, mesh_handle)

    def load_motion(idx):
        nonlocal \
            current_gts, \
            current_grs, \
            current_contacts, \
            current_dt, \
            current_n_frames, \
            scene_mesh_handle
        gts, grs, contacts, dt, n_frames = get_motion_frames(motion_data, idx)
        current_gts = gts
        current_grs = grs
        current_contacts = contacts
        current_dt = dt
        current_n_frames = n_frames
        frame_slider.max = n_frames - 1
        frame_slider.value = 0
        motion_label.value = motion_names[idx]

        # Load scene mesh
        if scene_mesh_handle is not None:
            for h in scene_mesh_handle:
                h.remove()
            scene_mesh_handle = None

        if scene_data is not None:
            scene_objs = load_scene_meshes(scene_data, idx, mesh_root)
            if scene_objs:
                scene_mesh_handle = []
                for j, (mesh, trans, rot) in enumerate(scene_objs):
                    rot_wxyz = quat_xyzw_to_wxyz(rot)
                    h = server.scene.add_mesh_trimesh(
                        f"/scene_object_{j}",
                        mesh=mesh,
                        wxyz=rot_wxyz,
                        position=trans,
                    )
                    scene_mesh_handle.append(h)

    def update_frame(frame_idx):
        if current_gts is None:
            return
        frame_idx = min(frame_idx, current_n_frames - 1)
        positions = current_gts[frame_idx]  # (23, 3)
        rotations = current_grs[frame_idx]  # (23, 4) xyzw

        has_contacts = current_contacts is not None and show_contacts.value
        contact_frame = current_contacts[frame_idx] if has_contacts else None

        for body_idx, (frame_handle, mesh_handle) in body_handles.items():
            pos = positions[body_idx]
            rot_xyzw = rotations[body_idx]
            rot_wxyz = quat_xyzw_to_wxyz(rot_xyzw)

            frame_handle.position = pos
            frame_handle.wxyz = rot_wxyz

            # Contact coloring: swap mesh color if in contact
            if has_contacts and contact_frame[body_idx]:
                color = [255, 60, 60]
            else:
                color = REGION_COLORS[BODY_REGION[body_idx]]
            # Update the local mesh color by replacing it
            local_mesh = body_local_meshes[body_idx].copy()
            local_mesh.visual.face_colors = color + [255]
            # Re-add the mesh (lightweight since geometry is small)
            frame_name = f"/skeleton/{BODY_NAMES[body_idx]}"
            server.scene.add_mesh_trimesh(f"{frame_name}/geom", mesh=local_mesh)

    # Initial load
    load_motion(0)
    update_frame(0)

    @motion_slider.on_update
    def _on_motion_change(event: viser.GuiEvent) -> None:
        nonlocal current_motion_idx
        current_motion_idx = int(motion_slider.value)
        load_motion(current_motion_idx)
        update_frame(0)

    @frame_slider.on_update
    def _on_frame_change(event: viser.GuiEvent) -> None:
        update_frame(int(frame_slider.value))

    @play_btn.on_click
    def _on_play(event: viser.GuiEvent) -> None:
        nonlocal playing
        playing = not playing

    @show_scene.on_update
    def _on_show_scene(event: viser.GuiEvent) -> None:
        if scene_mesh_handle is not None:
            for h in scene_mesh_handle:
                h.visible = show_scene.value

    # Playback loop
    try:
        last_time = time.time()
        while True:
            if playing and current_n_frames > 0:
                now = time.time()
                elapsed = now - last_time
                if elapsed >= current_dt / speed_slider.value:
                    last_time = now
                    next_frame = (int(frame_slider.value) + 1) % current_n_frames
                    frame_slider.value = next_frame
                    update_frame(next_frame)
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("Shutting down.")


if __name__ == "__main__":
    main()
