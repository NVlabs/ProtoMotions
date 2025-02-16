import os
from pathlib import Path
import joblib

import typer
import yaml
from process_hml3d_data import amass_to_amassx, is_valid_motion, ProcessingOptions
            

def main(
    existing_yaml_file: Path, 
    amass_data_path: Path,
    out_file: Path,
    datasets: str,  # Comma separated list of datasets to add
    humanoid_type: str = "smpl",
    motion_fps_path: Path = Path("data/yaml_files/motion_fps_amass.yaml"),
    occlusion_data_path: Path = Path("data/amass/amass_copycat_occlusion_v3.pkl"),
    ignore_occlusions: bool = False,
):
    assert len(datasets.split(",")) > 0, "Must have at least one dataset to add"
    
    dataset_list = datasets.split(",")
    
    existing_motions = set()
    with open(existing_yaml_file, "r") as f:
        existing_motion_file = yaml.load(f, Loader=yaml.SafeLoader)["motions"]
        for motion in existing_motion_file:
            existing_motions.add(motion["file"])
        
    with open(motion_fps_path, "r") as f:
        motion_fps = yaml.load(f, Loader=yaml.SafeLoader)
        
    occlusion_data = joblib.load(occlusion_data_path)

    options = ProcessingOptions(
        ignore_occlusions=ignore_occlusions,
    )

    last_motion_idx = existing_motion_file[-1]["idx"]
    if "sub_motions" in existing_motion_file[-1]:
        last_motion_idx = existing_motion_file[-1]["sub_motions"][-1]["idx"]

    for root, dirs, files in os.walk(amass_data_path):
        # Only consider subfolders that are in the datasets list
        if any(dataset in root for dataset in dataset_list):
            for file in files:
                # remove the main_motion_dir from the root
                save_root = root.replace(str(amass_data_path), "")
                # remove any leading slashes
                save_root = save_root.lstrip("/")

                file_rename = (
                    save_root
                    + "/"
                    + file.replace(".npz", ".npy")
                    .replace("-", "_")
                    .replace(" ", "_")
                    .replace("(", "_")
                    .replace(")", "_")
                )
                
                if humanoid_type == "smplx":
                    occlusion_key = ("_".join(file_rename.split("/")))[:-4]

                    occlusion_key = amass_to_amassx(occlusion_key)
                else:
                    occlusion_key = "-".join(["0"] + ["_".join(file_rename.split("/"))])[:-4]
            
                print(occlusion_key)
                print(file_rename)
                    
                if occlusion_key in occlusion_data:
                    this_motion_occlusion = occlusion_data[occlusion_key]
                else:
                    this_motion_occlusion = []
                    
                is_valid, fps_30_bound_frame = is_valid_motion(
                    this_motion_occlusion, occlusion_key, options
                )
                
                if file_rename in motion_fps and file_rename not in existing_motions and is_valid:
                    entry_dict = {
                        "idx": last_motion_idx + 1,
                        "file": file_rename,
                        "fps": motion_fps[file_rename],
                        "weight": 1.0
                    }
                    existing_motion_file.append(entry_dict)
                    last_motion_idx += 1


    file = open(out_file, "w")
    yaml.dump(existing_motion_file, file)
    file.close()


if __name__ == "__main__":
    typer.run(main)
