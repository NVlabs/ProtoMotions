from pathlib import Path
import numpy as np
import typer
import yaml
from poselib.skeleton.skeleton3d import SkeletonMotion
from typing import Optional


def get_root_offset(motion_file):
    print(motion_file)
    data = SkeletonMotion.from_file(str(motion_file))
    return data.global_translation[
        0, 0, :2
    ].tolist()  # Get the first frame's root translation


def main(
    samp_dir: Path,
    out_file: Optional[Path] = None,
    sampx_dir: Optional[Path] = None,
):
    assert (
        out_file is not None or sampx_dir is not None
    ), "Either out_file or sampx_dir must be provided"

    root_offsets = {}

    for motion_file in samp_dir.glob("*.npy"):
        motion_name = motion_file.stem
        root_offset = get_root_offset(motion_file)
        root_offsets[motion_name] = root_offset

    if sampx_dir:
        sampx_root_offsets = {}
        for motion_file in sampx_dir.glob("*.npy"):
            motion_name = motion_file.stem
            root_offset = get_root_offset(motion_file)
            sampx_root_offsets[motion_name] = root_offset

        for motion_name, samp_offset in root_offsets.items():
            if motion_name in sampx_root_offsets:
                sampx_offset = sampx_root_offsets[motion_name]
                if not np.allclose(samp_offset, sampx_offset, atol=1e-5):
                    print(
                        f"Offset mismatch for {motion_name}: SAMP offset = {samp_offset}, SAMPX offset = {sampx_offset}"
                    )
            else:
                print(f"Missing motion in SAMPX: {motion_name}")

    if out_file:
        with open(out_file, "w") as f:
            yaml.dump(root_offsets, f)

        print(f"Root offsets have been saved to {out_file}")


if __name__ == "__main__":
    typer.run(main)
