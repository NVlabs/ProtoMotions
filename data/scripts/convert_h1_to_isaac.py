# This code is adapted from https://github.com/zhengyiluo/phc/ for the H1 robot.

import os
import shutil
from pathlib import Path

import joblib
import torch
import typer
from tqdm import tqdm

from data.scripts.h1_humanoid_batch import Humanoid_Batch_H1
from data.scripts.h1_5_no_finger_humanoid_batch import Humanoid_Batch_H1_5_no_finger
from isaac_utils import torch_utils
from scipy.spatial.transform import Rotation as sRot


def main(
    amass_root_dir: Path,
    force_remake: bool = False,
    humanoid_type: str = "h1",
):
    if humanoid_type == "h1":
        mesh_parsers = Humanoid_Batch_H1(mjcf_file="phys_anim/data/assets/mjcf/h1.xml")
    elif humanoid_type == "h1_5_no_finger":
        mesh_parsers = Humanoid_Batch_H1_5_no_finger(
            extend_head=False, mjcf_file="phys_anim/data/assets/mjcf/h1_5_no_finger.xml"
        )
    else:
        raise ValueError(f"Unknown humanoid {humanoid_type}")

    folder_names = [
        f.path.split("/")[-1] for f in os.scandir(amass_root_dir) if f.is_dir()
    ]

    append_name = f"{humanoid_type}_isaac"

    for folder_name in folder_names:
        if append_name in folder_name:
            # Ignore folders where we store converted motions
            continue
        if not force_remake and f"{folder_name}-{append_name}" in folder_names:
            continue

        data_dir = amass_root_dir / folder_name
        output_dir = amass_root_dir / f"{folder_name}-{append_name}"

        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)

        print(f"Processing subset {folder_name}")
        os.mkdir(output_dir)

        files = [f for f in Path(data_dir).glob("**/*.pkl")]
        print(f"Processing {len(files)} files")

        files.sort()

        for filename in tqdm(files):
            print(f"Processing {filename}")
            if True:
                relative_path_dir = filename.relative_to(data_dir).parent
                relative_path_dir.mkdir(exist_ok=True, parents=True)

                outpath = (
                    output_dir
                    / relative_path_dir
                    / filename.name.replace(".pkl", ".npy")
                    .replace("-", "_")
                    .replace(" ", "_")
                    .replace("(", "_")
                    .replace(")", "_")
                )

                motion_data = joblib.load(filename)
                motion_data = motion_data[list(motion_data.keys())[0]]

                trans = torch_utils.to_torch(motion_data["root_trans_offset"]).clone()
                pose_aa = torch_utils.to_torch(motion_data["pose_aa"]).clone()

                dt = 1 / motion_data["fps"]

                B, J, N = pose_aa.shape

                curr_motion = mesh_parsers.fk_batch(
                    pose_aa[None,], trans[None,], return_full=True, dt=dt
                )

                curr_motion = {
                    k: v.squeeze() if torch.is_tensor(v) else v
                    for k, v in curr_motion.items()
                }

                torch.save(curr_motion, outpath)


if __name__ == "__main__":
    with torch.no_grad():
        typer.run(main)
