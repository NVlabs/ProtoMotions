# This code is adapted from https://github.com/zhengyiluo/phc/ for the H1 robot.

import os
from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable
import typer
from tqdm import tqdm

from data.scripts.h1.h1_humanoid_batch import Humanoid_Batch_H1
from data.scripts.h1.h1_5_no_finger_humanoid_batch import Humanoid_Batch_H1_5_no_finger
from scipy.spatial.transform import Rotation as sRot
from smpl_sim.smpllib.smpl_parser import SMPL_Parser
from data.scripts.h1.fit_h1_shape import create_h1_shape
from data.scripts.h1.fit_h1_shape import h1_joint_pick_idx, smpl_joint_pick_idx


def main(
    amass_root_dir: Path,
    force_remake: bool = False,
    humanoid_type: str = "h1",
):
    # Set device based on CUDA availability
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    folder_names = [
        f.path.split("/")[-1] for f in os.scandir(amass_root_dir) if f.is_dir()
    ]

    append_name = humanoid_type

    # Check if the optimized shape file exists
    shape_file = Path(f"data/{humanoid_type}/shape_optimized_v1.pt")
    if not shape_file.exists():
        print(
            f"Optimized shape file not found. Running grad_fit_h1_shape.py for {humanoid_type}..."
        )
        if humanoid_type == "h1":
            create_h1_shape()
        else:
            raise NotImplementedError(
                f"The humanoid {humanoid_type} is not yet supported."
            )

        if not shape_file.exists():
            raise FileNotFoundError(
                f"Failed to generate optimized shape file for {humanoid_type}"
            )
        else:
            print(f"Successfully generated optimized shape file for {humanoid_type}")

    shape_new, scale = torch.load(shape_file)
    shape_new = shape_new.to(device)
    scale = scale.detach().to(device)

    number_of_retargeted_files = 0

    for folder_name in folder_names:
        if (
            append_name in folder_name
            or "retarget" in folder_name
            or "smpl" in folder_name
        ):
            # Ignore folders where we store converted motions
            continue

        data_dir = amass_root_dir / folder_name
        output_dir = amass_root_dir / f"{folder_name}-{append_name}"

        print(f"Processing subset {folder_name}")
        os.makedirs(output_dir, exist_ok=True)

        files = [
            f
            for f in Path(data_dir).glob("**/*.npz")
            if (f.name != "shape.npz" and "stagei.npz" not in f.name)
        ]
        print(f"Processing {len(files)} files")

        files.sort()

        for filename in tqdm(files):
            # try:
            if True:
                with torch.no_grad():
                    relative_path_dir = filename.relative_to(data_dir).parent
                    relative_path_dir.mkdir(exist_ok=True, parents=True)

                    outpath = (
                        output_dir
                        / relative_path_dir
                        / filename.name.replace(".npz", ".npy")
                        .replace("-", "_")
                        .replace(" ", "_")
                        .replace("(", "_")
                        .replace(")", "_")
                    )
                    # Check if the output file already exists
                    if not force_remake and outpath.exists():
                        print(f"Skipping {filename} as it already exists.")
                        continue
                    
                    number_of_retargeted_files += 1
                    
                    print(f"Processing {filename}")
                    
                    # Create the output directory if it doesn't exist
                    os.makedirs(output_dir / relative_path_dir, exist_ok=True)

                    motion_data = np.load(filename)

                    amass_pose = motion_data["poses"]
                    amass_trans = motion_data["trans"]
                    if "mocap_framerate" in motion_data:
                        mocap_fr = motion_data["mocap_framerate"]
                    else:
                        mocap_fr = motion_data["mocap_frame_rate"]

                    skip = int(mocap_fr // 30)

                    dt = 1.0 / 30

                    pose_aa = torch.tensor(amass_pose[::skip]).float().to(device)
                    amass_trans = torch.tensor(amass_trans[::skip]).float().to(device)

                    batch_size = pose_aa.shape[0]
                    
                    if batch_size <= 1:
                        # Bad motion!
                        number_of_retargeted_files -= 1  # Undo the count
                        continue

                    pose_aa = torch.cat(
                        [
                            pose_aa[:, :66],
                            torch.zeros(
                                (batch_size, 6),
                                dtype=torch.float,
                                device=device,
                            ),
                        ],
                        dim=1,
                    )  # TODO: need to extract correct handle rotations instead of zero

                    smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")
                    smpl_parser_n.to(device)

                    _, joints = smpl_parser_n.get_joints_verts(pose_aa, shape_new, amass_trans)
                    root_pos = joints[:, 0:1]
                    joints = (joints - joints[:, 0:1]) * scale.detach() + root_pos

                    offset = joints[:, 0] - amass_trans
                    root_trans_offset = (amass_trans + offset).clone()

                    gt_root_rot = (
                        torch.from_numpy(
                            (
                                sRot.from_rotvec(pose_aa.cpu().numpy()[:, :3])
                                * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
                            ).as_rotvec()
                        )
                        .float()
                        .to(device)
                    )

                    dof_pos = torch.zeros((1, batch_size, 19, 1)).to(device)

                dof_pos_new = Variable(dof_pos, requires_grad=True)
                optimizer_pose = torch.optim.Adadelta([dof_pos_new], lr=100)

                with torch.no_grad():
                    if humanoid_type == "h1":
                        mesh_parsers = Humanoid_Batch_H1(
                            mjcf_file="phys_anim/data/assets/mjcf/h1.xml",
                            extend_hand=True,
                            extend_head=False,
                            device=device,
                        )
                    elif humanoid_type == "h1_5_no_finger":
                        mesh_parsers = Humanoid_Batch_H1_5_no_finger(
                            extend_head=False,
                            mjcf_file="phys_anim/data/assets/mjcf/h1_5_no_finger.xml",
                            device=device,
                        )
                    else:
                        raise ValueError(f"Unknown humanoid {humanoid_type}")

                for iteration in range(500):
                    pose_aa_h1_new = torch.cat(
                        [
                            gt_root_rot[None, :, None],
                            mesh_parsers.rotation_axis * dof_pos_new,
                            torch.zeros((1, batch_size, 2, 3)).to(device),
                        ],
                        axis=2,
                    ).to(device)

                    fk_return = mesh_parsers.fk_batch(
                        pose_aa_h1_new, root_trans_offset[None,], dt=dt
                    )

                    diff = (
                        fk_return["global_translation"][:, :, h1_joint_pick_idx]
                        - joints[:, smpl_joint_pick_idx]
                    )
                    loss_g = diff.norm(dim=-1).mean()
                    loss = loss_g

                    optimizer_pose.zero_grad()
                    loss.backward()
                    optimizer_pose.step()

                    if iteration % 50 == 0:
                        print(f"{iteration} {loss.item() * 1000}")

                    dof_pos_new.data.clamp_(
                        mesh_parsers.joints_range[:, 0, None],
                        mesh_parsers.joints_range[:, 1, None],
                    )

                with torch.no_grad():
                    pose_aa_h1_new = torch.cat(
                        [
                            gt_root_rot[None, :, None],
                            mesh_parsers.rotation_axis * dof_pos_new,
                            torch.zeros((1, batch_size, 2, 3)).to(device),
                        ],
                        axis=2,
                    )
                    fk_return = mesh_parsers.fk_batch(
                        pose_aa_h1_new, root_trans_offset[None,], dt=dt
                    )

                    min_height = fk_return.global_translation[..., 2].min().item()
                    fk_return.global_translation[..., 2] -= min_height - 0.08

                    curr_motion = {
                        k: v.squeeze().detach().cpu() if torch.is_tensor(v) else v
                        for k, v in fk_return.items()
                    }

                    torch.save(curr_motion, outpath)
                    
                    if number_of_retargeted_files >= 5:
                        return
            # except Exception as e:
            #     print(f"Error processing {filename}")
            #     print(f"Error: {e}")
            #     print(f"Line: {e.__traceback__.tb_lineno}")
            #     continue


if __name__ == "__main__":
    typer.run(main)
