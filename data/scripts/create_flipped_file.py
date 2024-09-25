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

import copy
from pathlib import Path

import typer
import yaml


def main(in_file: Path):
    final_yaml_dict_format = {"motions": []}
    num_motions = 0
    num_sub_motions = 0
    total_time = 0

    motions = yaml.load(open(in_file, "r"), Loader=yaml.FullLoader)["motions"]

    for motion in motions:
        num_motions += 1
        new_motion = copy.deepcopy(motion)
        new_motion["idx"] = num_sub_motions
        for sub_motion in new_motion["sub_motions"]:
            sub_motion["idx"] = num_sub_motions
            num_sub_motions += 1
            total_time += sub_motion["timings"]["end"] - sub_motion["timings"]["start"]

        final_yaml_dict_format["motions"].append(new_motion)

        flipped_item_dict = copy.deepcopy(motion)
        flipped_item_dict["file"] = motion["file"].replace(".npy", "_flipped.npy")

        flipped_item_dict["idx"] = num_sub_motions
        for sub_motion in flipped_item_dict["sub_motions"]:
            sub_motion["labels"] = [
                label.replace("left", "rrrttt")
                .replace("right", "left")
                .replace("rrrttt", "right")
                for label in sub_motion["labels"]
            ]
            sub_motion["labels"] = [
                label.replace("clockwise", "rrrttt")
                .replace("counterclockwise", "clockwise")
                .replace("rrrttt", "counterclockwise")
                for label in sub_motion["labels"]
            ]
            sub_motion["idx"] = num_sub_motions
            num_sub_motions += 1

        final_yaml_dict_format["motions"].append(flipped_item_dict)

    file = open(str(in_file).replace(".yaml", "_flipped.yaml"), "w")
    yaml.dump(final_yaml_dict_format, file)
    file.close()

    print(
        f"Total of {num_motions * 2} motions, and {num_sub_motions * 2} sub-motions, spanning {total_time * 2 / 60} minutes."
    )


if __name__ == "__main__":
    typer.run(main)
