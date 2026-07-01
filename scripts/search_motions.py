# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Search packaged motion files (.pt) by name substring.

Usage:
    python scripts/search_motions.py chunk_00.pt "running"
    python scripts/search_motions.py chunk_00.pt "jump" --full-path
"""

import argparse
import os

import torch


def main():
    parser = argparse.ArgumentParser(description="Search motion names in a .pt file")
    parser.add_argument("pt_file", help="Path to the .pt motion file")
    parser.add_argument("query", help="Substring to search for (case-insensitive)")
    parser.add_argument(
        "--full-path", action="store_true", help="Show full paths instead of filenames"
    )
    args = parser.parse_args()

    data = torch.load(args.pt_file, weights_only=False, map_location="cpu")
    motion_files = data["motion_files"]
    query = args.query.lower()

    matches = []
    for i, path in enumerate(motion_files):
        name = path if args.full_path else os.path.basename(path)
        if query in name.lower():
            matches.append((i, name))

    if matches:
        print(f"Found {len(matches)} / {len(motion_files)} motions matching '{args.query}':\n")
        for idx, name in matches:
            print(f"  {idx:>6d}  {name}")
    else:
        print(f"No motions matching '{args.query}' in {len(motion_files)} entries.")


if __name__ == "__main__":
    main()
