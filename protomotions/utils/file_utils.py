# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
