import sys
import argparse
import os
import numpy as np
import torch
import glob
import smplx
import yaml
from scipy.spatial.transform import Rotation as R
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
import trimesh
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

motion_to_object = {
    "armchair": "Armchairs",
    "chair": "StraightChairs",
    "highstool": "HighStools",
    "reebokstep": "LowStools",
    "sofa": "Sofas",
    "table": "Tables",
    "lie_down": "LargeSofas",
}


def process_sequences(cfg):
    samp_path = cfg.samp_path

    all_seqs = glob.glob(os.path.join(samp_path, "*.pkl"))

    total_sequences = len(all_seqs)
    if cfg.max_seqs:
        total_sequences = min(total_sequences, cfg.max_seqs)

    motions_data = []
    scenes_data = []

    sequence_idx = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Processing sequences...", total=total_sequences)

        for sequence in all_seqs:
            print(f"Processing {sequence}")
            motion_info, scene_info = process_sequence(cfg, sequence, sequence_idx)
            if motion_info is None:
                continue
            motions_data.append(motion_info)
            scenes_data.append(scene_info)
            sequence_idx += 1
            progress.update(task, advance=1)

            if cfg.max_seqs and sequence_idx >= cfg.max_seqs:
                break

    if cfg.out_path:
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to the yaml_files directory
        yaml_files_dir = os.path.join(current_dir, "..", "yaml_files")

        # Ensure the directory exists
        os.makedirs(yaml_files_dir, exist_ok=True)

        # Save motion data
        motions_yaml_path = os.path.join(yaml_files_dir, "samp_motions.yaml")
        with open(motions_yaml_path, "w") as f:
            yaml.dump({"motions": motions_data}, f)

        # Save scene data
        scenes_yaml_path = os.path.join(yaml_files_dir, "samp.yaml")
        with open(scenes_yaml_path, "w") as f:
            yaml.dump({"scenes": scenes_data}, f)


def process_sequence(cfg, sequence, sequence_idx):
    # Load and process SAMP sequence data
    with open(sequence, "rb") as f:
        seq_data = pickle.load(f, encoding="latin1")

    # Extract relevant information from seq_data
    poses = seq_data["pose_est_fullposes"]
    trans = seq_data["pose_est_trans"]
    betas = seq_data["shape_est_betas"][:10]
    gender = "neutral"
    betas[:] = 0
    fps = seq_data["mocap_framerate"]

    T = poses.shape[0]  # Number of frames

    # Create SMPL-X model
    smplx_model = smplx.create(
        model_path=cfg.model_path,
        model_type="smplx",
        gender=gender,
        num_betas=betas.shape[0],
        batch_size=T,
    )

    body_pose = poses[:, 3:66]
    hand_poses = poses[:, 66:114]  # 45 for each hand
    left_hand_poses = hand_poses[:, :24]  # 24
    right_hand_poses = hand_poses[:, 24:]  # 24

    # Process the parameters through the SMPL-X model
    smplx_output = smplx_model(
        betas=torch.tensor(betas[None, :]).float().expand(T, -1),
        global_orient=torch.tensor(poses[:, :3]).float(),  # 3
        body_pose=torch.tensor(body_pose).float(),  # 63
        left_hand_pose=torch.tensor(left_hand_poses).float(),  # 24
        right_hand_pose=torch.tensor(right_hand_poses).float(),  # 24
        # transl=torch.tensor(trans).float(),
        return_full_pose=True,
        use_pca=False,
    )

    # Extract full pose and vertices
    full_pose = smplx_output.full_pose.detach().cpu().numpy()
    vertices = smplx_output.vertices.detach().cpu().numpy()

    # Determine the object category
    sequence_name = os.path.basename(sequence).split(".")[0]
    object_category = None
    for key, value in motion_to_object.items():
        if key in sequence_name:
            object_category = value
            break

    if object_category is None:
        print(
            f"Warning: Could not determine object category for sequence {sequence_name}"
        )

    # Load object mesh
    object_mesh = None
    if object_category:
        object_dir = os.path.join(cfg.object_asset_path, "SAMP", object_category)
        obj_files = glob.glob(os.path.join(object_dir, "*.obj"))
        if obj_files:
            object_mesh_path = obj_files[0]  # Pick the first obj file
            object_trimesh = trimesh.load(object_mesh_path)
            object_mesh = Mesh(
                vertices=object_trimesh.vertices,
                faces=object_trimesh.faces,
                vc=colors["yellow"],
            )

    # Visualization
    if cfg.visualize:
        mv = MeshViewer(offscreen=False)

        # set the camera pose
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = R.from_euler(
            "xyz", [80, -15, 0], degrees=True
        ).as_matrix()
        camera_pose[:3, 3] = np.array([-0.5, -4.0, 1.5])
        mv.update_camera_pose(camera_pose)

        # Create body mesh
        body_mesh = Mesh(
            vertices=vertices[0],
            faces=smplx_model.faces,
            vc=colors["pink"],
            smooth=True,
        )

        # Visualize every 5th frame
        for frame in range(0, T, 5):
            body_mesh.vertices = vertices[frame]
            meshes = [body_mesh]

            if object_mesh is not None:
                meshes.append(object_mesh)

            mv.set_static_meshes(meshes)

            # Optionally, you can add a small delay to slow down the visualization
            # import time
            # time.sleep(0.1)

    if cfg.out_path:
        # Create folder structure
        motions_dir = os.path.join(cfg.out_path, "motions", "SAMP")
        os.makedirs(motions_dir, exist_ok=True)

        # Save motion data
        motion_filename = f"{os.path.basename(os.path.dirname(sequence))}_{os.path.basename(sequence).split('.')[0]}.npz"
        motion_path = os.path.join(motions_dir, motion_filename)
        np.savez(
            motion_path,
            poses=full_pose,
            trans=trans,
            betas=betas,
            gender=gender,
            mocap_framerate=fps,
        )

        # Create motion_info and scene_info
        scene_name = f"samp_{os.path.basename(os.path.dirname(sequence))}_{os.path.basename(sequence).split('.')[0]}"

        motion_info = {
            "file": os.path.relpath(
                motion_path, os.path.join(cfg.out_path, "motions/SAMP")
            ),
            "idx": sequence_idx,
            "sub_motions": [
                {
                    "action_type": "samp_motion",
                    "idx": sequence_idx,
                    "supported_scenes": [scene_name],
                    "weight": 1.0,
                }
            ],
            "fps": fps,
        }

        scene_info = {
            "id": scene_name,
            "objects": (
                [
                    {
                        "is_static": True,
                        "path": f"protomotions/data/assets/scenes/train/SAMP/{object_category}/{os.path.basename(object_mesh_path)}",
                        "object_options": {
                            "vhacd_enabled": True,
                            "vhacd_params": {
                                "max_convex_hulls": 20,
                                "max_num_vertices_per_ch": 64,
                                "resolution": 300000,
                            },
                            "fix_base_link": True,
                            "density": 1000,
                        },
                    }
                ]
                if object_category
                else []
            ),
            "replications": 1,
        }
    else:
        motion_info = None
        scene_info = None

    return motion_info, scene_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAMP-process")

    parser.add_argument(
        "--samp-path",
        required=True,
        type=str,
        help="The path to the downloaded SAMP data",
    )

    parser.add_argument(
        "--model-path",
        required=True,
        type=str,
        help="The path to the folder containing smplx models",
    )

    parser.add_argument(
        "--out-path",
        type=str,
        help="The path to save the processed data and YAML files",
    )

    parser.add_argument(
        "--max-seqs",
        type=int,
        default=None,
        help="The maximum number of sequences to process",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the sequences",
    )

    parser.add_argument(
        "--grab-code-path",
        required=True,
        type=str,
        help="The path to the GRAB code folder containing the 'tools' directory",
    )

    parser.add_argument(
        "--object-asset-path",
        required=True,
        type=str,
        help="The path to the folder containing object assets",
    )

    args = parser.parse_args()

    # Add GRAB code path to sys.path
    sys.path.append(args.grab_code_path)

    # Now import the tools
    from tools.objectmodel import ObjectModel
    from tools.meshviewer import Mesh, MeshViewer, colors
    from tools.utils import parse_npz
    from tools.utils import params2torch
    from tools.utils import to_cpu
    from tools.utils import euler
    from tools.cfg_parser import Config

    assert (
        args.out_path is not None or args.visualize
    ), "Please specify either --out-path or --visualize"

    cfg = {
        "samp_path": args.samp_path,
        "model_path": args.model_path,
        "out_path": args.out_path,
        "max_seqs": args.max_seqs,
        "visualize": args.visualize,
        "object_asset_path": args.object_asset_path,
    }

    cfg = Config(**cfg)
    process_sequences(cfg)
