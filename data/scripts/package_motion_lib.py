import sys
import os
from pathlib import Path
sys.path.insert(0, "/home/cizinsky/ProtoMotions")

import torch
import typer
import yaml
import tempfile
import math
from hydra.utils import get_class
from hydra import compose, initialize

from protomotions.utils.motion_lib import MotionLib
from protomotions.simulator.base_simulator.config import RobotConfig

from omegaconf import OmegaConf, ListConfig

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver("sqrt", lambda x: math.sqrt(float(x)))
OmegaConf.register_new_resolver("sum", lambda x: sum(x))
OmegaConf.register_new_resolver("ceil", lambda x: math.ceil(x))
OmegaConf.register_new_resolver("int", lambda x: int(x))
OmegaConf.register_new_resolver("len", lambda x: len(x))
OmegaConf.register_new_resolver("sum_list", lambda lst: sum(lst))
OmegaConf.register_new_resolver("len_or_int_value", lambda lst: len(lst) if isinstance(lst, ListConfig) else int(lst))


def main(
        motion_file: Path,
        amass_data_path: Path,
        outpath: Path,
        humanoid_type: str = "smpl",
        num_data_splits: int = None,
):
    config_path = "../../protomotions/config/robot"

    with initialize(version_base=None, config_path=config_path, job_name="test_app"):
        cfg = compose(config_name=humanoid_type)

    key_body_ids = torch.tensor(
        [
            cfg.robot.body_names.index(key_body_name)
            for key_body_name in cfg.robot.key_bodies
        ],
        dtype=torch.long,
    )

    # Process the robot config into a RobotConfig object
    robot_config: RobotConfig = RobotConfig.from_dict(cfg.robot)

    print("Creating motion state")
    motion_files = []
    if num_data_splits is not None:
        # Motion file is a yaml file
        # Just load the yaml and break it into num_data_splits
        # Save each split as a separate file
        with open(os.path.join(os.getcwd(), motion_file), "r") as f:
            motions = yaml.load(f, Loader=yaml.SafeLoader)["motions"]
        num_motions = len(motions)
        split_size = num_motions // num_data_splits
        for i in range(num_data_splits):
            if i == num_data_splits - 1:  # make sure we get all the remaining motions
                split_motions = motions[i * split_size:]
            else:
                split_motions = motions[i * split_size: (i + 1) * split_size]

            motion_idx = 0
            for motion in split_motions:
                motion["idx"] = motion_idx
                if "sub_motions" in motion:
                    for sub_motion in motion["sub_motions"]:
                        sub_motion["idx"] = motion_idx
                        motion_idx += 1
                else:
                    motion_idx += 1

            split_name = motion_file.with_name(
                motion_file.stem + f"_{i}" + motion_file.suffix
            )
            with open(split_name, "w") as f:
                yaml.dump({"motions": split_motions}, f)

            motion_files.append(
                (
                    str(split_name),
                    outpath.with_name(outpath.stem + f"_{i}" + outpath.suffix),
                )
            )
    else:
        motion_files.append((motion_file, outpath))

    for motion_file, outpath in motion_files:
        # Open and edit the motion file
        with open(motion_file, "r") as f:
            motion_data = yaml.safe_load(f)

        # Edit file paths
        for motion in motion_data["motions"]:
            motion["file"] = str(amass_data_path.resolve() / motion["file"])

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
        ) as temp_file:
            yaml.dump(motion_data, temp_file)
            temp_file_path = temp_file.name

        # Use the temporary file for MotionLib
        cfg.motion_lib.motion_file = temp_file_path

        MotionLibClass = get_class(cfg.motion_lib._target_)
        motion_lib_params = {}
        for key, value in cfg.motion_lib.items():
            if key != "_target_":
                motion_lib_params[key] = value

        mlib: MotionLib = MotionLibClass(
            robot_config=robot_config,
            key_body_ids=key_body_ids,
            device="cpu",
            skeleton_tree=None,
            **motion_lib_params
        )

        print("Saving motion state")

        with open(outpath, "wb") as file:
            torch.save(mlib.state, file)

        # Remove the temporary file
        os.unlink(temp_file_path)


if __name__ == "__main__":
    typer.run(main)
