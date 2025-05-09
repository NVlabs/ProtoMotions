import json
from pathlib import Path
from typing import List

import yaml
from omegaconf import OmegaConf


def load_yaml(fname: Path):
    with open(fname) as file:
        data = yaml.load(file, yaml.CLoader)
    return data


def load_motions(fname: Path) -> List:
    motions = load_yaml(fname)["motions"]
    motion_names = []
    for motion in motions:
        if "sub_motions" in motion:
            for sub_motion in motion["sub_motions"]:
                if "hml3d_id" in sub_motion:
                    motion_names.append(sub_motion["hml3d_id"])
                else:
                    # legacy
                    motion_names.append(sub_motion["timings"]["labels"]["seg_id"])
        else:
            motion_names.append(motion["file"])
    return motion_names


def load_omegaconf(fname: Path):
    return OmegaConf.create(load_yaml(fname))


def save_yaml(obj, fname: Path):
    with open(fname, "w") as file:
        yaml.dump(obj, file)


def save_motions(motions, fname: Path):
    obj = {"motions": motions}
    save_yaml(obj, fname)


def load_json(fname: Path):
    with open(fname) as file:
        data = json.load(file)
    return data
