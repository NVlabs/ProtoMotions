# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
