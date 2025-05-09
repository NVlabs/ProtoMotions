import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import typer
import yaml
from tqdm import tqdm

from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState

HML3D_FPS = 20


def amass_to_amassx(file_path):
                file_path = file_path.replace("_poses", "_stageii")
                file_path = file_path.replace("SSM_synced", "SSM")
                file_path = file_path.replace("MPI_HDM05", "HMD05")
                file_path = file_path.replace("MPI_mosh", "MoSh")
                file_path = file_path.replace("MPI_Limits", "PosePrior")
                file_path = file_path.replace("TCD_handMocap", "TCDHands")
                file_path = file_path.replace("Transitions_mocap", "Transitions")
                file_path = file_path.replace("DFaust_67", "DFaust")
                file_path = file_path.replace("BioMotionLab_NTroje", "BMLrub")
                return file_path
            

@dataclass
class ProcessingOptions:
    ignore_occlusions: bool
    occlusion_bound: int = 0
    occlusion: int = 0


def fix_motion_fps(motion, dur):
    true_fps = motion.local_rotation.shape[0] / dur

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        motion.skeleton_tree,
        motion.local_rotation,
        motion.root_translation,
        is_local=True,
    )
    new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=true_fps)

    return new_motion


def is_valid_motion(
    occlusion_data: dict,
    motion_name: str,
    options: ProcessingOptions,
):
    if not options.ignore_occlusions and len(occlusion_data) > 0:
        issue = occlusion_data["issue"]
        if (issue == "sitting" or issue == "airborne") and "idxes" in occlusion_data:
            bound = occlusion_data["idxes"][
                0
            ]  # This bounded is calculated assuming 30 FPS.....
            if bound < 10:
                options.occlusion_bound += 1
                print("bound too small", motion_name, bound)
                return False, 0
            else:
                return True, bound
        else:
            options.occlusion += 1
            print("issue irrecoverable", motion_name, issue)
            return False, 0

    return True, None


def main(
    outfile: Path,
    amass_data_path: Path,
    text_dir: Path = Path("data/hml3d/texts"),
    csv_file: Path = Path("data/hml3d/index.csv"),
    hml3d_file: Path = Path("data/hml3d/train_val.txt"),
    motion_fps_path: Path = Path("data/yaml_files/motion_fps_amass.yaml"),
    occlusion_data_path: Path = Path("data/amass/amass_copycat_occlusion_v3.pkl"),
    humanoid_type: str = "smpl",
    dataset: str = "",
    max_length_seconds: Optional[int] = None,  # 90
    min_length_seconds: Optional[float] = 0.5,
    ignore_occlusions: bool = False,
):
    """
    We need the babel file to get the duration of the clip
    to adjust the fps.
    """

    num_too_long = 0
    num_too_short = 0
    total_time = 0
    total_motions = 0
    total_sub_motions = 0

    # load csv file
    df = pd.read_csv(csv_file)
    # load text file and iterate line by line
    hml3d_indices = []
    with open(hml3d_file) as f:
        # readh file line by line and store integers in the line
        for line in f:
            entry = line
            # ignore mirrored files
            if entry.startswith("M"):
                continue
            # remove line ends
            entry = entry.strip()
            hml3d_indices.append(int(entry))

    occlusion_data = joblib.load(occlusion_data_path)

    motion_fps_dict = yaml.load(open(motion_fps_path, "r"), Loader=yaml.FullLoader)

    output_motions = {}

    options = ProcessingOptions(
        ignore_occlusions=ignore_occlusions,
    )

    for k, hml3d_idx in enumerate(tqdm(hml3d_indices)):
        path = (
            df["source_path"][hml3d_idx][12:]
            .replace(".npz", ".npy")
            .replace("-", "_")
            .replace(" ", "_")
            .replace("(", "_")
            .replace(")", "_")
        )

        if dataset not in path and dataset != "":
            continue

        path_parts = path.split(os.path.sep)
        path_parts[0] = path_parts[0] + "-" + humanoid_type
        key = os.path.join(*(path_parts))

        if humanoid_type == "smplx":
            occlusion_key = ("_".join(path.split("/")))[:-4]
            key = amass_to_amassx(key)
            path = key.replace("-smplx", "")

            occlusion_key = amass_to_amassx(occlusion_key)
        else:
            occlusion_key = "-".join(["0"] + ["_".join(path.split("/"))])[:-4]

        if not os.path.exists(f"{amass_data_path}/{key}"):
            continue

        if occlusion_key in occlusion_data:
            this_motion_occlusion = occlusion_data[occlusion_key]
        else:
            this_motion_occlusion = []

        if path not in motion_fps_dict:
            raise Exception(f"{path} not in motion_fps_dict.")
        else:
            motion_fps = motion_fps_dict[path]

        is_valid, fps_30_bound_frame = is_valid_motion(
            this_motion_occlusion, occlusion_key, options
        )
        if not is_valid:
            continue

        rid = hml3d_idx

        # get row as a dict
        row = df.iloc[rid].to_dict()

        new_name = row["new_name"]
        label_path = (text_dir / new_name).with_suffix(".txt")
        raw_labels = label_path.read_text().strip().split("\n")

        processed_labels = []
        for raw_label in raw_labels:
            label = raw_label.split("#")[0].strip()
            if label.endswith("."):
                label = label[:-1]
            processed_labels.append(label)

        # extract the motion
        raw_start_frame = row["start_frame"]
        if fps_30_bound_frame is not None:
            raw_end_frame = min(
                row["end_frame"], int(np.floor(fps_30_bound_frame * 1.0 / 30 * 20))
            )
        else:
            raw_end_frame = row["end_frame"]

        start_time = raw_start_frame / HML3D_FPS
        end_time = raw_end_frame / HML3D_FPS
        length_seconds = end_time - start_time
        if max_length_seconds is not None and length_seconds > max_length_seconds:
            num_too_long += 1
            continue

        if length_seconds < min_length_seconds:
            num_too_short += 1
            continue

        if key not in output_motions:
            output_motions[key] = []
            total_motions += 1

        output_motions[key].append(
            {
                "start": start_time,
                "end": end_time,
                "fps": motion_fps,
                "hml3d_id": rid,
                "labels": processed_labels,
            }
        )

        total_time += end_time - start_time
        total_sub_motions += 1

    yaml_dict_format = {"motions": []}
    num_motions = 0
    num_sub_motions = 0
    for key, value in output_motions.items():
        if humanoid_type == "smplx":
            key = key.replace("_poses.npy", "_stageii.npy").replace(
                "-smpl/", "-smplx/"
            )  # change filenames to match the AMASS-X naming convention.
        item_dict = {
            "file": key,
            "fps": value[0]["fps"],
            "sub_motions": [],
            "idx": num_sub_motions,
        }
        num_motions += 1
        for sub_motion in value:
            item_dict["sub_motions"].append(
                {
                    "timings": {"start": sub_motion["start"], "end": sub_motion["end"]},
                    "weight": 1.0,
                    "idx": num_sub_motions,
                    "hml3d_id": sub_motion["hml3d_id"],
                    "labels": sub_motion["labels"],
                }
            )
            num_sub_motions += 1

        yaml_dict_format["motions"].append(item_dict)

    print(f"Saving {len(output_motions)} motions to {outfile}")
    print(
        f"Total of {num_motions} motions, and {num_sub_motions} sub-motions, equaling to {total_time / 60} minutes of motion."
    )
    print(f"Num too long: {num_too_long}")
    print(f"Num too short: {num_too_short}")
    print(
        f"Num occluded: {options.occlusion}, occluded_bound: {options.occlusion_bound}"
    )

    with open(outfile, "w") as file:
        yaml.dump(yaml_dict_format, file)


if __name__ == "__main__":
    typer.run(main)
