# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the native Metatron MuJoCo model once, outside the control loop."""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from tools.metatron_v1_mujoco_model import patch_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--start-prone", action="store_true", help="simulation recovery fixture only"
    )
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError("output must differ from the source MJCF")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite prepared MJCF: {args.output}")
    prepared, directory = patch_model(args.source, pd_mode="implicit")
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.start_prone:
            tree = ET.parse(prepared)
            pelvis = tree.find("./worldbody/body[@name='pelvis']")
            if pelvis is None:
                raise ValueError("prepared G1 XML has no pelvis root body")
            pelvis.set("pos", "0 0 0.25")
            pelvis.set("quat", "0 0 1 0")
            tree.write(args.output, encoding="unicode", xml_declaration=True)
        else:
            shutil.copyfile(prepared, args.output)
    finally:
        directory.cleanup()
    print(f"native MuJoCo model ready: {args.output}")


if __name__ == "__main__":
    main()
