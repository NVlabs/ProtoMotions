# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert a serialized SOMA SceneLib to +X forward without changing geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from data.scripts.from_soma_to_anatomical_soma.soma23_to_anatomical_soma_retargeting import (  # noqa: E402
    CANONICAL_FRAME,
    SOMA_TO_CANONICAL,
    canonical_soma_scenes,
    save_verified,
)


def same_scene_data(left, right):
    """Exact equality for SceneLib's nested containers and heightfield tensors."""
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            same_scene_data(left[k], right[k]) for k in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            same_scene_data(a, b) for a, b in zip(left, right)
        )
    return left == right


def convert(source: Path, destination: Path):
    if destination.exists() or destination.resolve() == source.resolve():
        raise ValueError("Refusing to overwrite a scene output or its source")
    data = torch.load(source, weights_only=True, map_location="cpu")
    output = canonical_soma_scenes(data)
    assets = []
    for scene in output["original_scenes"]:
        for obj in scene["objects"]:
            if obj["type"] == "MeshSceneObject":
                path = Path(obj["object_path"])
                path = path if path.is_absolute() else source.parent / path
                path = path.resolve(strict=True)
                obj["object_path"] = str(path)
                assets.append(str(path))
    # Heightfield scenes contain tensors: Python dict equality is ambiguous
    # for multi-element tensors. Check the whole nested structure exactly.
    save_verified(output, destination, lambda loaded: same_scene_data(loaded, output))
    destination.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "coordinate_system": CANONICAL_FRAME,
                "basis": SOMA_TO_CANONICAL.tolist(),
                "object_orientation_transform": "C @ R; local object geometry unchanged",
                "referenced_assets": sorted(set(assets)),
                "num_original_scenes": output["num_original_scenes"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.input, args.output)
